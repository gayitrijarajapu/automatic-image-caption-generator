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
- Branch: `main`
- Main file path: `app.py`

Required files:

- `app.py`
- `requirements.txt`
- `yolov8m.pt`
- `README.md`
- `frontend/index.html`

Select Python 3.11 in Streamlit Cloud's advanced settings. Keep the frontend
folder in the repository: it contains the supplied HTML interface connected to
the Python models. The default theme is Plum (`#7c35c8` on `#f6f1f8`).
Uploads and the built-in sample run real YOLO/BLIP inference. Changing language
reuses cached analysis. Listen uses speech voices available in your browser.
