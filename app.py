import os
from collections import Counter
from pathlib import Path
from io import BytesIO
import hashlib

os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st
from PIL import Image
from deep_translator import GoogleTranslator
from gtts import gTTS

@st.cache_data(show_spinner=False, ttl=86400, max_entries=256)
def translate_caption(caption, lang):
    if lang == "en":
        return caption
    return GoogleTranslator(source="en", target=lang).translate(caption)

# Load models once
@st.cache_resource(show_spinner=False)
def load_models():
    from ultralytics import YOLO
    from transformers import BlipProcessor, BlipForConditionalGeneration

    model_path = Path(__file__).parent / "yolov8m.pt"
    yolo_model = YOLO(str(model_path))
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return yolo_model, processor, blip_model

st.set_page_config(page_title="Frame | Image Caption Generator", layout="wide")

# Hide Streamlit chrome (toolbar/header/footer), including Deploy option.
st.markdown(
    """
    <style>
    [data-testid="stToolbar"],
    [data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        visibility: hidden;
        height: 0;
        position: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap');
:root { --canvas:#F2F4F8; --surface:#FFFFFF; --surface2:#F7F8FC; --ink:#16213E; --muted:#5D6883; --line:#DFE4EE; --accent:#2F4BFF; --soft:#EEF1FF; --accentline:#AEBBFF; }
.stApp { background:var(--canvas); color:var(--ink); color-scheme:light; font-family:'Public Sans',sans-serif; }
.block-container { max-width:1180px; padding:1.25rem 2rem 3rem; }
p, label, button, input, select { font-family:'Public Sans',sans-serif; letter-spacing:0; }
h1,h2,h3 { font-family:'Bricolage Grotesque',sans-serif !important; color:var(--ink); letter-spacing:0; }
h1 { font-size:44px !important; line-height:1.07 !important; max-width:22ch; }
h3 { font-size:20px !important; }
.topbar { font-family:'Bricolage Grotesque',sans-serif; font-weight:600; border-bottom:1px solid var(--line); padding-bottom:16px; margin-bottom:20px; color:var(--ink); }
.topbar span { display:inline-grid; place-items:center; width:30px; height:30px; border-radius:8px; background:var(--accent); color:white; margin-right:10px; }
.st-key-upload_panel,.st-key-preview_panel { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:24px; box-shadow:0 1px 2px #16213e0a,0 18px 36px -24px #16213e2e; }
[data-testid="stFileUploaderDropzone"] { min-height:280px; display:flex; flex-direction:column; justify-content:center; gap:16px; background:var(--soft); border:1.5px dashed var(--accentline); border-radius:14px; text-align:center; }
[data-testid="stFileUploaderDropzone"] * { color:var(--muted); }
[data-testid="stFileUploaderDropzone"] button { background:var(--surface); border:1px solid var(--accentline); color:var(--accent); border-radius:10px; }
[data-testid="stBaseButton-primary"] { height:46px; background:var(--accent); color:white; border:0; border-radius:10px; font-weight:600; }
[data-testid="stBaseButton-primary"]:hover { background:#2238D6; color:white; }
[data-testid="stBaseButton-primary"]:disabled { background:#D9DEEB; color:#6F7994; }
[data-testid="stBaseButton-secondary"] { color:var(--accent); background:var(--surface); border:1px solid var(--accentline); border-radius:10px; }
[data-baseweb="select"] > div { background:var(--surface); color:var(--ink); border-color:var(--line); min-height:46px; border-radius:10px; }
[data-baseweb="select"] svg { fill:var(--muted); }
[role="listbox"],[role="option"] { background:var(--surface); color:var(--ink); }
[data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"] { color:var(--muted); }
[data-testid="stImage"] img { max-height:360px; object-fit:contain; border-radius:12px; }
[data-testid="stExpander"],[data-baseweb="tab"],[data-testid="stMetricValue"],[data-testid="stMetricLabel"] { color:var(--ink); }
hr { border-color:var(--line); }
.caption-block { border-left:3px solid var(--accent); padding-left:16px; font-family:'Bricolage Grotesque',sans-serif; font-size:26px; line-height:1.25; overflow-wrap:anywhere; }
@media(max-width:640px) { .block-container { padding:1rem; } h1 { font-size:32px !important; } .st-key-upload_panel,.st-key-preview_panel { padding:18px; } }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="topbar"><span aria-hidden="true">&#8982;</span>Image Caption Generator</div>', unsafe_allow_html=True)
st.title("Turn any image into words, in any language.")

languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr",
}

left_col, right_col = st.columns(2, gap="medium")
left = left_col.container(key="upload_panel")
right = right_col.container(key="preview_panel")
with left:
    st.subheader("Your image")
    uploaded_file = st.file_uploader("Image file", type=["jpg", "png", "jpeg", "webp"], label_visibility="collapsed")

image = None
source_label = ""

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("This image could not be opened. Choose another JPG or PNG.")
        st.stop()
    source_label = uploaded_file.name

with right:
    lang_name = st.selectbox("Translate caption to", list(languages.keys()))
    lang = languages[lang_name]
    generate = st.button("Generate caption", type="primary", icon=":material/auto_awesome:", use_container_width=True, disabled=image is None)



if image is None:
    st.session_state.pop("caption_result", None)
    with right:
        st.divider()
        st.subheader("Your results appear here")
        st.caption("No image selected")

if image is not None:
    with left:
        st.image(image, caption=source_label, use_container_width=True)
        st.caption(f"{image.width} x {image.height} px")
    result_key = (hashlib.sha256(uploaded_file.getvalue()).hexdigest(), lang)
    if st.session_state.get("result_key") != result_key:
        st.session_state.pop("caption_result", None)
    if generate:
        try:
            with st.spinner("Loading AI models. The first run downloads about 1 GB; later runs use the cache."):
                yolo_model, processor, blip_model = load_models()
        except Exception as exc:
            st.error("Could not load the AI models. Check your internet connection and try again.")
            st.exception(exc)
            st.stop()

        with st.spinner("⏳ Processing..."):

            # YOLO Detection
            results = yolo_model(image)
            boxes = results[0].boxes
            object_names = results[0].names

            detected = []
            confidence_rows = []
            for box in boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                label = object_names[cls_idx]
                detected.append(label)
                confidence_rows.append({"Object": label, "Confidence": f"{conf:.0%}"})

            # Count objects
            object_count = Counter(detected)

            # Draw YOLO bounding boxes
            annotated = results[0].plot()
            annotated_rgb = annotated[:, :, ::-1]

            # BLIP Caption
            inputs = processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Translation
            voice_lang = lang
            try:
                translated = translate_caption(caption, lang)
            except Exception:
                translated = caption
                voice_lang = "en"
                st.warning("Translation is temporarily unavailable. Showing the English caption; try translation again later.")

            # Voice
            audio = BytesIO()
            try:
                gTTS(translated, lang=voice_lang).write_to_fp(audio)
            except Exception:
                audio = None
                st.warning("Voice output is temporarily unavailable. Your caption is still available.")

        st.session_state.result_key = result_key
        st.session_state.caption_result = dict(caption=caption, translated=translated,
            voice_lang=voice_lang, audio=audio.getvalue() if audio is not None else None,
            annotated=annotated_rgb, counts=dict(object_count), rows=confidence_rows)

    result = st.session_state.get("caption_result")
    if result:
        with right:
            st.divider()
            st.caption("ENGLISH CAPTION")
            from html import escape
            st.markdown('<div class="caption-block">' + escape(result["caption"]) + '</div>', unsafe_allow_html=True)
            if result["voice_lang"] != "en":
                st.caption(lang_name.upper())
                st.write(result["translated"])
        with right:
            st.subheader(":material/graphic_eq: Voice output")
            if result["audio"]:
                st.audio(result["audio"], format="audio/mp3")
            else:
                st.caption("Voice output unavailable")
            st.download_button("Download caption", data=result["translated"],
                file_name="caption.txt", icon=":material/download:", use_container_width=True)
        with right.expander("Object detection and image insights"):
            total, categories, people = st.columns(3)
            total.metric("Objects", sum(result["counts"].values()))
            categories.metric("Categories", len(result["counts"]))
            people.metric("People", result["counts"].get("person", 0))
            preview, details = st.tabs(["Detection preview", "Object details"])
            with preview:
                st.image(result["annotated"], use_container_width=True)
            with details:
                if result["rows"]:
                    st.dataframe(result["rows"], hide_index=True, use_container_width=True)
                else:
                    st.caption("No objects detected")
