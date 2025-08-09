# LEGOClip

Create short LEGO-styled clips from a text description or by transforming an uploaded video.

## Features
- Text-to-LEGO clip: generate a base LEGO image using Stable Diffusion + LEGO LoRA, then animate it (Stable Video Diffusion if available; fallback to a smooth zoom).
- Video-to-LEGO: stylize an uploaded clip frame-by-frame using Img2Img.
- Content safety: blocks unsafe prompts and checks uploaded/generated frames.

## Setup
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
2. Install dependencies (ensure you have `ffmpeg` on your system):
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional, macOS with Apple Silicon) If using MPS, installs are the same; models will run on MPS automatically.

## Run
```bash
python app.py
```

This will launch a local Gradio app. Open the printed URL in your browser.

## Notes
- Models used:
  - `runwayml/stable-diffusion-v1-5` (text-to-image, img2img)
  - `Kontext-Style/LEGO_lora` (style LoRA)
  - `stabilityai/stable-video-diffusion-img2vid-xt` (optional, if available)
- If SVD is unavailable or fails, the app falls back to a simple Ken Burns animation.
- Keep uploaded clips short (≤ ~3s) for responsiveness.

