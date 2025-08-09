import os
import io
import re
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
import imageio

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from contextlib import nullcontext
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor

try:
    from diffusers import StableVideoDiffusionPipeline
    SVD_AVAILABLE = True
except Exception:
    SVD_AVAILABLE = False


# -----------------------
# Device and dtype setup
# -----------------------
def select_device() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # float16 on MPS can be unstable; prefer float32
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


DEVICE, DTYPE = select_device()


# -----------------------
# Content moderation
# -----------------------
BLOCKED_PATTERNS = [
    # Sexual content
    r"\bporn\b",
    r"\bsex\b",
    r"\bsexual\b",
    r"\bnude\b",
    r"\bnudity\b",
    r"\bnsfw\b",
    r"\berotic\b",
    r"\bexplicit\b",
    # Violence / gore
    r"\bgore\b",
    r"\bbehead(ing)?\b",
    r"\bdismember\w*\b",
    r"\bbloodbath\b",
    r"\brape\b",
    # Hate / harassment
    r"\bslur\b",
    r"\bracist\b",
    r"\bkill\b\s+\w+",
    r"\blynch\b",
    r"\bgenocide\b",
]


_BLOCKED_REGEX = [re.compile(pat, flags=re.IGNORECASE) for pat in BLOCKED_PATTERNS]


def is_text_safe(text: str) -> Tuple[bool, Optional[str]]:
    if not text or not text.strip():
        return False, "Prompt is empty."
    for rx in _BLOCKED_REGEX:
        if rx.search(text):
            # Report the pattern in a human-readable way
            pattern_str = rx.pattern.replace("\\b", "").replace("\\w+", "<word>")
            return False, f"Blocked by content policy (found pattern: '{pattern_str}')."
    return True, None


# -----------------------
# Safety checker for images
# -----------------------
_safety_checker: Optional[StableDiffusionSafetyChecker] = None
_clip_processor: Optional[CLIPImageProcessor] = None


def load_safety_checker():
    global _safety_checker, _clip_processor
    if _safety_checker is None:
        _safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cpu")
    if _clip_processor is None:
        _clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )


def images_are_safe(pil_images: List[Image.Image]) -> bool:
    load_safety_checker()
    images_np = (
        np.stack([np.array(im.convert("RGB")) for im in pil_images]).astype(np.float32) / 255.0
    )
    inputs = _clip_processor(images=pil_images, return_tensors="pt")
    with torch.no_grad():
        _, has_nsfw = _safety_checker(images=images_np, clip_input=inputs.pixel_values)
    return not any(bool(x) for x in (has_nsfw or []))


# -----------------------
# Model pipelines
# -----------------------
_txt2img: Optional[StableDiffusionPipeline] = None
_img2img: Optional[StableDiffusionImg2ImgPipeline] = None
_svd: Optional["StableVideoDiffusionPipeline"] = None
_upscaler = None
_clip_rank_model: Optional[CLIPModel] = None
_clip_rank_processor: Optional[CLIPProcessor] = None


def load_pipelines():
    global _txt2img, _img2img, _svd
    if _txt2img is None:
        _txt2img = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        try:
            _txt2img.load_lora_weights("Kontext-Style/LEGO_lora")
        except Exception:
            pass
        _txt2img = _txt2img.to(DEVICE)

    if _img2img is None:
        _img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        try:
            _img2img.load_lora_weights("Kontext-Style/LEGO_lora")
        except Exception:
            pass
        _img2img = _img2img.to(DEVICE)

    # Do not load SVD at startup. We'll lazy-load it on first use to speed up app boot.


def ensure_svd_loaded():
    global _svd
    if _svd is None and SVD_AVAILABLE:
        try:
            _svd = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=DTYPE,
                use_safetensors=True,
            )
            _svd = _svd.to(DEVICE)
        except Exception:
            _svd = None


def ensure_upscaler_loaded():
    global _upscaler
    if _upscaler is None:
        try:
            from diffusers import StableDiffusionUpscalePipeline
        except Exception:
            _upscaler = None
            return
        try:
            _upscaler = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=DTYPE,
            ).to(DEVICE)
        except Exception:
            _upscaler = None


def ensure_clip_ranker_loaded():
    global _clip_rank_model, _clip_rank_processor
    if _clip_rank_model is None or _clip_rank_processor is None:
        _clip_rank_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        _clip_rank_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# -----------------------
# Utility helpers
# -----------------------
def augment_prompt(user_prompt: str) -> str:
    lego_style = (
        "LEGO style, brick-built characters and environments, minifigure style, "
        "toy photography, colorful plastic bricks, studded surfaces, cinematic lighting"
    )
    return f"{user_prompt}, {lego_style}"


def make_video(frames: List[Image.Image], fps: int = 14) -> bytes:
    # Write MP4 to bytes buffer
    buf = io.BytesIO()
    temp_path = "_tmp_output.mp4"
    imageio.mimwrite(temp_path, [np.array(f.convert("RGB")) for f in frames], fps=fps, quality=8)
    with open(temp_path, "rb") as f:
        data = f.read()
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return data


def ken_burns_frames(image: Image.Image, frames: int = 24) -> List[Image.Image]:
    # Simple fallback animation: slow zoom-in
    w, h = image.size
    crop_scale_start, crop_scale_end = 0.9, 0.75
    result = []
    for i in range(frames):
        t = i / max(frames - 1, 1)
        scale = crop_scale_start * (1 - t) + crop_scale_end * t
        cw, ch = int(w * scale), int(h * scale)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        frame = image.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)
        result.append(frame)
    return result


# -----------------------
# Core generation routines
# -----------------------
def generate_lego_clip_from_text(
    prompt: str,
    negative_prompt: str = (
        "nsfw, nude, explicit, low quality, blurry, deformed, watermark, text, logo, "
        "worst quality, lowres, extra limbs, missing fingers, malformed, disfigured"
    ),
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    fps: int = 14,
    frames: int = 24,
    use_svd: bool = False,
    best_of: int = 1,
    upscale_x4: bool = False,
) -> Tuple[str, Image.Image]:
    safe, reason = is_text_safe(prompt)
    if not safe:
        raise gr.Error(reason)

    load_pipelines()

    full_prompt = augment_prompt(prompt)
    generator = torch.Generator(device=DEVICE.type)
    if seed is not None:
        generator = generator.manual_seed(int(seed))

    amp_ctx = torch.autocast(device_type="cuda") if DEVICE.type == "cuda" else nullcontext()
    # Generate multiple candidates and rank by CLIP text-image similarity
    num_candidates = max(1, int(best_of))
    candidate_images: List[Image.Image] = []
    with amp_ctx:
        for i in range(num_candidates):
            gen = generator if seed is not None else None
            if gen is None:
                gen = torch.Generator(device=DEVICE.type).manual_seed(torch.seed())
            img_i: Image.Image = _txt2img(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                height=512,
                width=512,
            ).images[0]
            candidate_images.append(img_i)

    # Rank
    if len(candidate_images) > 1:
        ensure_clip_ranker_loaded()
        inputs = _clip_rank_processor(text=[prompt], images=candidate_images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            out = _clip_rank_model(**inputs)
            image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims = (image_embeds @ text_embeds.T).squeeze(-1)
            best_idx = int(torch.argmax(sims).item())
        img = candidate_images[best_idx]
    else:
        img = candidate_images[0]

    # Optional safety check on result image
    if not images_are_safe([img]):
        raise gr.Error("Generated image failed safety checks.")

    # Optional upscaling for sharper video
    if upscale_x4:
        ensure_upscaler_loaded()
        if _upscaler is not None:
            with amp_ctx:
                img = _upscaler(prompt="highly detailed, sharp, clean edges", image=img).images[0]

    # Animate
    video_frames: List[Image.Image]
    if use_svd:
        ensure_svd_loaded()
    if use_svd and _svd is not None:
        try:
            with amp_ctx:
                vid = _svd(
                    image=img,
                    num_inference_steps=20,
                    num_frames=frames,
                    decode_chunk_size=8,
                    generator=generator,
                ).frames[0]
            video_frames = [Image.fromarray(f) for f in vid]
        except Exception:
            video_frames = ken_burns_frames(img, frames=frames)
    else:
        video_frames = ken_burns_frames(img, frames=frames)

    mp4_bytes = make_video(video_frames, fps=fps)
    return (mp4_bytes, img)


def stylize_video_to_lego(
    video_path: str,
    prompt_hint: str = "",
    strength: float = 0.55,
    guidance_scale: float = 6.5,
    max_frames: int = 64,
    fps_out: Optional[int] = None,
) -> bytes:
    if prompt_hint:
        safe, reason = is_text_safe(prompt_hint)
        if not safe:
            raise gr.Error(reason)

    load_pipelines()

    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps_in = int(meta.get("fps", 14))
    except Exception:
        fps_in = 14

    target_fps = fps_out or min(fps_in, 16)

    # Read and sample frames
    frames_np: List[np.ndarray] = []
    for idx, frame in enumerate(reader):
        if idx >= max_frames:
            break
        frames_np.append(frame)
    reader.close()

    if not frames_np:
        raise gr.Error("No frames found in the uploaded video.")

    # Safety check sampled frames (every 8th)
    sample_indices = list(range(0, len(frames_np), max(1, len(frames_np)//8)))
    sample_images = [Image.fromarray(frames_np[i]) for i in sample_indices]
    if not images_are_safe(sample_images):
        raise gr.Error("Uploaded video appears unsafe.")

    # Process frames with img2img
    processed_frames: List[Image.Image] = []
    lego_hint = augment_prompt("")
    full_prompt = (prompt_hint + ", " + lego_hint).strip(", ")
    amp_ctx = torch.autocast(device_type="cuda") if DEVICE.type == "cuda" else nullcontext()
    for frame_np in frames_np:
        base_img = Image.fromarray(frame_np).convert("RGB").resize((512, 512), Image.LANCZOS)
        with amp_ctx:
            out = _img2img(
                prompt=full_prompt or "LEGO style scene, keep composition and structure",
                image=base_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=25,
            ).images[0]
        processed_frames.append(out)

    # Write video
    mp4_bytes = make_video(processed_frames, fps=target_fps)
    return mp4_bytes


# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks(title="LEGO Clip Studio") as demo:
    gr.Markdown("""
    ## LEGO Clip Studio
    - Describe a scene to generate a short LEGO-styled clip
    - Or upload a video clip to transform it into LEGO style
    - Safety filters block disallowed content
    """)

    with gr.Tab("Describe a Scene"):
        with gr.Row():
            prompt = gr.Textbox(label="Describe the scene", placeholder="e.g., A heroic minifigure swings between LEGO skyscrapers at sunset")
        with gr.Accordion("Advanced", open=False):
            negative = gr.Textbox(value="nsfw, nude, explicit, low quality, blurry, deformed, watermark, text, logo", label="Negative prompt")
            steps = gr.Slider(10, 60, value=30, step=1, label="Diffusion steps")
            guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="Guidance scale")
            seed = gr.Number(value=None, label="Seed (optional)")
            fps = gr.Slider(6, 24, value=14, step=1, label="FPS")
            nframes = gr.Slider(12, 48, value=24, step=1, label="Frames")
            use_svd = gr.Checkbox(value=False, label="Use Stable Video Diffusion (slower first run)")
            best_of = gr.Slider(1, 6, value=3, step=1, label="Best-of images (pick best match)")
            upscale = gr.Checkbox(value=True, label="Upscale x4 for sharper video")
        gen_btn = gr.Button("Generate LEGO Clip")
        out_video = gr.Video(label="Generated Clip", format="mp4")
        out_image = gr.Image(label="Base Frame", type="pil")

        def _gen_text(prompt, negative, steps, guidance, seed, fps, nframes, use_svd, best_of, upscale):
            mp4, img = generate_lego_clip_from_text(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                seed=int(seed) if seed is not None else None,
                fps=int(fps),
                frames=int(nframes),
                use_svd=bool(use_svd),
                best_of=int(best_of),
                upscale_x4=bool(upscale),
            )
            # Gradio Video expects a temp file path; write and return path
            tmp_path = "_gen_text.mp4"
            with open(tmp_path, "wb") as f:
                f.write(mp4)
            return tmp_path, img

        gen_btn.click(_gen_text, [prompt, negative, steps, guidance, seed, fps, nframes, use_svd, best_of, upscale], [out_video, out_image])

    with gr.Tab("Upload a Clip"):
        in_video = gr.Video(label="Upload a short clip (<= ~3s recommended)")
        with gr.Accordion("Advanced", open=False):
            prompt_hint = gr.Textbox(label="Optional style hint", placeholder="e.g., cinematic lighting, city street at night")
            strength = gr.Slider(0.2, 0.9, value=0.55, step=0.05, label="Stylization strength")
            guidance2 = gr.Slider(1.0, 12.0, value=6.5, step=0.5, label="Guidance scale")
            max_frames = gr.Slider(8, 128, value=64, step=1, label="Max frames to process")
            fps_out = gr.Slider(6, 24, value=14, step=1, label="Output FPS")
        stylize_btn = gr.Button("Transform to LEGO Style")
        out_video2 = gr.Video(label="Stylized Clip", format="mp4")

        def _stylize(video, prompt_hint, strength, guidance2, max_frames, fps_out):
            if video is None:
                raise gr.Error("Please upload a video file.")
            mp4 = stylize_video_to_lego(
                video_path=video,
                prompt_hint=prompt_hint or "",
                strength=float(strength),
                guidance_scale=float(guidance2),
                max_frames=int(max_frames),
                fps_out=int(fps_out),
            )
            tmp_path = "_stylized.mp4"
            with open(tmp_path, "wb") as f:
                f.write(mp4)
            return tmp_path

        stylize_btn.click(_stylize, [in_video, prompt_hint, strength, guidance2, max_frames, fps_out], [out_video2])


if __name__ == "__main__":
    load_pipelines()
    demo.launch(inbrowser=True)
