---
title: Image Caption Generator
sdk: docker
app_port: 7860
---

# Image Caption Generator

The supplied HTML interface is served by web_app.py, with real YOLO object
detection and BLIP captions. Translation falls back to English when unavailable.
Voice playback uses the browser's installed voices; language availability varies.

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python web_app.py
```

Open http://localhost:8502. Stop with Ctrl+C.
Models download on first generation and remain cached.
The original Streamlit interface remains in app.py.

## Deploy

Create a Hugging Face Space with Docker SDK and upload:
web_app.py, index.html, requirements.txt, yolov8m.pt, Dockerfile, README.md.
The Docker container serves port 7860. Model downloads require internet access.

The sample button loads an illustration; generation analyzes it with the real
models, so results may differ from the original HTML's demonstration text.
