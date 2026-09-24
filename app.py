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
:root {
  --canvas:#f6f1f8; --surface:#FFFFFF; --surface2:#FAF7FC; --ink:#2A1636;
  --muted:#6A5877; --line:#E5DCEC; --accent:#7c35c8; --accent-hover:#5F26A0;
  --soft:#F1E8FA; --accentline:#C9A9EC; --disabled-bg:#E2D9EA; --disabled-ink:#7C6C89;
  --shadow:0 1px 2px rgba(42,22,54,.04),0 18px 36px -24px rgba(42,22,54,.2);
}
.stApp { background:var(--canvas); color:var(--ink); color-scheme:light; font-family:'Public Sans',sans-serif; }
.block-container { max-width:1180px; padding:1.25rem 2rem 3rem; }
p, label, button, input, select, textarea { font-family:'Public Sans',sans-serif; letter-spacing:0; }
h1,h2,h3 { font-family:'Bricolage Grotesque',sans-serif !important; color:var(--ink); letter-spacing:0; }
h1 { font-size:46px !important; line-height:1.07 !important; max-width:22ch; margin-bottom:.75rem !important; }
h3 { font-size:20px !important; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; border-bottom:1px solid var(--line); padding-bottom:16px; margin-bottom:36px; color:var(--ink); }
.brand { display:flex; align-items:center; gap:.65rem; font-family:'Bricolage Grotesque',sans-serif; font-weight:600; font-size:1.05rem; }
.brand-icon { display:inline-grid; place-items:center; width:30px; height:30px; border-radius:8px; background:var(--accent); color:white; }
.top-note { color:var(--muted); font-size:.88rem; }
.swatches { display:flex; gap:.55rem; align-items:center; }
.swatch { width:28px; height:28px; border-radius:50%; border:1px solid var(--line); background:linear-gradient(135deg,var(--bg) 50%,var(--ac) 50%); box-shadow:0 0 0 3px var(--canvas),0 0 0 5px transparent; }
.swatch.active { box-shadow:0 0 0 3px var(--canvas),0 0 0 5px var(--accent); }
.hero-sub { color:var(--muted); font-size:1.05rem; line-height:1.55; max-width:58ch; margin:0 0 2rem; }
.st-key-upload_panel,.st-key-preview_panel,.st-key-results_panel {
  background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:24px; box-shadow:var(--shadow);
}
.card-head { display:flex; gap:.8rem; align-items:flex-start; margin-bottom:1rem; }
.card-icon { flex:none; display:grid; place-items:center; width:40px; height:40px; border-radius:10px; background:var(--soft); color:var(--accent); border:1px solid var(--accentline); font-weight:700; }
.card-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.35rem; line-height:1.15; font-weight:600; margin:0; color:var(--ink); }
.card-sub { color:var(--muted); font-size:.94rem; margin:.2rem 0 0; }
[data-testid="stFileUploaderDropzone"] { min-height:300px; display:flex; flex-direction:column; justify-content:center; gap:16px; background:var(--soft); border:1.5px dashed var(--accentline); border-radius:14px; text-align:center; }
[data-testid="stFileUploaderDropzone"] * { color:var(--muted); }
[data-testid="stFileUploaderDropzone"] button { background:var(--surface); border:1px solid var(--accentline); color:var(--accent); border-radius:10px; font-weight:600; }
[data-testid="stBaseButton-primary"] { height:46px; background:var(--accent); color:white; border:0; border-radius:10px; font-weight:600; }
[data-testid="stBaseButton-primary"]:hover { background:var(--accent-hover); color:white; }
[data-testid="stBaseButton-primary"]:disabled { background:var(--disabled-bg); color:var(--disabled-ink); }
[data-testid="stBaseButton-secondary"] { color:var(--accent); background:var(--surface); border:1px solid var(--accentline); border-radius:10px; }
[data-baseweb="select"] > div { background:var(--surface); color:var(--ink); border-color:var(--line); min-height:46px; border-radius:10px; }
[data-baseweb="select"] svg { fill:var(--muted); }
[role="listbox"],[role="option"] { background:var(--surface); color:var(--ink); }
[data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"] { color:var(--muted); }
[data-testid="stImage"] img { max-height:360px; object-fit:contain; border-radius:12px; border:1px solid var(--line); background:var(--surface2); }
[data-testid="stExpander"],[data-baseweb="tab"],[data-testid="stMetricValue"],[data-testid="stMetricLabel"] { color:var(--ink); }
hr { border-color:var(--line); margin:1.35rem 0; }
.steps { list-style:none; padding:0; margin:0; display:grid; gap:.85rem; }
.steps li { display:flex; align-items:center; gap:.8rem; color:var(--muted); font-size:.98rem; }
.steps .n { flex:none; width:1.7rem; height:1.7rem; border-radius:50%; display:inline-grid; place-items:center; font-size:.85rem; font-weight:600; color:var(--muted); border:1px solid var(--line); background:var(--surface2); }
.steps li.done { color:var(--ink); }
.steps li.done .n { background:var(--accent); border-color:var(--accent); color:white; }
.caption-block { border-left:3px solid var(--accent); padding:.1rem 0 .1rem 1rem; margin-bottom:1.4rem; }
.caption-label { color:var(--muted); font-size:.85rem; font-weight:500; margin:0 0 .35rem; }
.caption-text { font-family:'Bricolage Grotesque',sans-serif; font-size:1.75rem; line-height:1.25; font-weight:500; color:var(--ink); overflow-wrap:anywhere; margin:0; }
.translation-block { border-left:3px solid var(--line); padding:.1rem 0 .1rem 1rem; margin-bottom:1.2rem; }
.note { display:flex; gap:.55rem; margin-top:1.2rem; padding:.7rem .85rem; border-radius:10px; background:var(--surface2); border:1px solid var(--line); color:var(--muted); font-size:.84rem; line-height:1.45; }
.privacy { color:var(--muted); font-size:.82rem; line-height:1.5; margin:1rem 0 0; }
@media(max-width:720px) { .top-note,.swatches { display:none; } }
@media(max-width:640px) { .block-container { padding:1rem; } h1 { font-size:32px !important; } .st-key-upload_panel,.st-key-preview_panel,.st-key-results_panel { padding:18px; } }
</style>""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="topbar">
      <div class="brand"><span class="brand-icon">⌖</span><span>Image Caption Generator</span></div>
      <div class="swatches" aria-hidden="true">
        <span class="top-note">YOLOv8 finds the objects. BLIP writes the caption.</span>
        <span class="swatch" style="--bg:#F2F4F8;--ac:#2F4BFF"></span>
        <span class="swatch" style="--bg:#EEF4F1;--ac:#0B7A54"></span>
        <span class="swatch active" style="--bg:#f6f1f8;--ac:#7c35c8"></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Turn any image into words, in any language.")
st.markdown('<p class="hero-sub">Upload a photo to detect the objects in it, get a written caption, translate it, and hear it read aloud.</p>', unsafe_allow_html=True)

languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}

left_col, right_col = st.columns(2, gap="medium")
left = left_col.container(key="upload_panel")
right = right_col.container(key="preview_panel")
with left:
    st.markdown(
        '<div class="card-head"><span class="card-icon">1</span><div><p class="card-title">Upload an image</p><p class="card-sub">Choose an image to generate a caption.</p></div></div>',
        unsafe_allow_html=True,
    )
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
    st.markdown(
        '<div class="card-head"><span class="card-icon">2</span><div><p class="card-title">Caption controls</p><p class="card-sub">Select a language and start the AI analysis.</p></div></div>',
        unsafe_allow_html=True,
    )
    lang_name = st.selectbox("Translate caption to", list(languages.keys()))
    lang = languages[lang_name]
    generate = st.button("Generate caption", type="primary", icon=":material/auto_awesome:", use_container_width=True, disabled=image is None)



if image is None:
    st.session_state.pop("caption_result", None)
    with right:
        st.divider()
        st.markdown(
            """
            <p class="card-title">Your results appear here</p>
            <ol class="steps">
              <li><span class="n">1</span>Upload a JPG, PNG or WebP image</li>
              <li><span class="n">2</span>Choose a language for the translation</li>
              <li><span class="n">3</span>Select Generate caption</li>
            </ol>
            """,
            unsafe_allow_html=True,
        )

if image is not None:
    with left:
        st.image(image, caption=source_label, use_container_width=True)
        st.caption(f"{image.width} x {image.height} px")
        st.markdown('<p class="privacy">Images are processed on the app server. Caption text may be sent to translation and voice services.</p>', unsafe_allow_html=True)
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
    if not result:
        with right:
            st.divider()
            st.markdown(
                """
                <p class="card-title">Your results appear here</p>
                <ol class="steps">
                  <li class="done"><span class="n">1</span>Image uploaded</li>
                  <li class="done"><span class="n">2</span>Language selected</li>
                  <li><span class="n">3</span>Select Generate caption</li>
                </ol>
                """,
                unsafe_allow_html=True,
            )
    if result:
        with right:
            st.divider()
            from html import escape
            st.markdown(
                '<div class="caption-block"><p class="caption-label">Caption</p><p class="caption-text">' +
                escape(result["caption"]) + '</p></div>',
                unsafe_allow_html=True,
            )
            if result["voice_lang"] != "en":
                st.markdown(
                    '<div class="translation-block"><p class="caption-label">' + escape(lang_name) +
                    '</p><p>' + escape(result["translated"]) + '</p></div>',
                    unsafe_allow_html=True,
                )
        with right:
            st.subheader(":material/graphic_eq: Voice output")
            if result["audio"]:
                st.audio(result["audio"], format="audio/mp3")
            else:
                st.caption("Voice output unavailable")
            st.download_button("Download caption", data=result["translated"],
                file_name="caption.txt", icon=":material/download:", use_container_width=True)
            st.markdown('<p class="note">AI-generated captions may contain mistakes.</p>', unsafe_allow_html=True)
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
