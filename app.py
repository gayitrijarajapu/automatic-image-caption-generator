import base64
import logging
from io import BytesIO
from pathlib import Path
from threading import Lock

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
from deep_translator import GoogleTranslator


@st.cache_resource(show_spinner=False)
def load_models():
    from ultralytics import YOLO
    from transformers import BlipProcessor, BlipForConditionalGeneration
    detector = YOLO(str(Path(__file__).parent / "yolov8m.pt"))
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    captioner = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").eval()
    return detector, processor, captioner, Lock()


@st.cache_data(show_spinner=False, ttl=86400, max_entries=256)
def translate_caption(caption, language):
    if language == "en":
        return caption
    return GoogleTranslator(source="en", target=language).translate(caption)


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def analyze_image(image_bytes):
    import torch
    image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes)))
    if image.width * image.height > 20_000_000:
        raise ValueError("Image dimensions are too large")
    image = image.convert("RGB")
    detector, processor, captioner, lock = load_models()
    # Model instances are shared across sessions, so serialize inference.
    with lock, torch.inference_mode():
        prediction = detector(image, verbose=False)[0]
        inputs = processor(image, return_tensors="pt")
        tokens = captioner.generate(**inputs, max_new_tokens=60)
        caption = processor.decode(tokens[0], skip_special_tokens=True)
    colors = ["#2F4BFF", "#0B7F8C", "#C2410C", "#7c35c8", "#0B7A54"]
    objects = []
    for i, box in enumerate(prediction.boxes):
        x1, y1, x2, y2 = box.xyxyn[0].tolist()
        objects.append({"label": prediction.names[int(box.cls[0])],
                        "score": round(float(box.conf[0]) * 100),
                        "color": colors[i % len(colors)],
                        "box": [100*x1, 100*y1, 100*(x2-x1), 100*(y2-y1)]})
    return {"caption": caption, "objects": objects}


st.set_page_config(page_title="Image Caption Generator", layout="wide")
st.markdown("""<style>
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu {display:none;}
.stApp {background:#f6f1f8;}
.block-container {max-width:none;padding:0;}
iframe {border:none;display:block;width:100%;}
</style>""", unsafe_allow_html=True)
caption_ui = components.declare_component("caption_ui", path=str(Path(__file__).parent / "frontend"))
request = caption_ui(response=st.session_state.get("response"), key="caption_ui", default=None)
if isinstance(request, dict) and request.get("id") != st.session_state.get("handled_request"):
    request_id = request.get("id")
    st.session_state.handled_request = request_id
    try:
        language = request.get("lang", "en")
        if language not in {"en", "hi", "te", "ta", "es", "fr", "de", "ja"}:
            raise ValueError("Unsupported language")
        encoded = request["image"]
        if len(encoded) > 28_000_000 or not encoded.startswith("data:image/jpeg;base64,"):
            raise ValueError("Invalid image")
        image_bytes = base64.b64decode(encoded.split(",", 1)[1], validate=True)
        result = dict(analyze_image(image_bytes))
        result.update(id=request_id, voice_lang=language, warning="")
        try:
            result["translated"] = translate_caption(result["caption"], language)
        except Exception:
            logging.exception("Caption translation failed")
            result.update(translated=result["caption"], voice_lang="en",
                          warning="Translation is temporarily unavailable. Showing the English caption.")
        st.session_state.response = result
    except Exception:
        logging.exception("Image analysis failed")
        st.session_state.response = {"id": request_id,
            "error": "Could not analyze the image. Please retry. If it continues, check the server logs."}
    st.rerun()
