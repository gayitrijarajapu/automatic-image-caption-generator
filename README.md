---
title: Image Caption Generator
sdk: streamlit
app_file: app.py
---

# Image Caption Generator

AI image caption generator built with Streamlit. The app detects objects with
YOLOv8, generates captions with BLIP, translates captions, and creates voice
output.

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Deploy

Deploy on Streamlit Community Cloud with:

- Repository: `gayitrijarajapu/automatic-image-caption-generator`
- Branch: `version-2`
- Main file path: `app.py`

Required files:

- `app.py`
- `requirements.txt`
- `yolov8m.pt`
- `README.md`
