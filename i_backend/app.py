import os
from collections import Counter

os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from deep_translator import GoogleTranslator
from gtts import gTTS

# Load models once
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8m.pt")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return yolo_model, processor, blip_model

yolo_model, processor, blip_model = load_models()

st.set_page_config(page_title="Image Caption Generator", layout="centered")

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

st.markdown(
    """
    <style>
    .big-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 4px;
    }
    .sub-title {
        text-align: center;
        color: #4b5563;
        margin-bottom: 18px;
    }
    .note-box {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 10px 12px;
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-title">🧠 AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Image -> YOLO -> BLIP -> Translation -> Voice</p>', unsafe_allow_html=True)

with st.container():
    st.markdown(
        '<div class="note-box">We combined object detection and caption generation to improve contextual understanding.</div>',
        unsafe_allow_html=True,
    )

st.write("")

languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr",
}

st.subheader("📥 Input Source")
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

image = None
source_label = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source_label = "📸 Uploaded Image"

if image is not None:
    st.image(image, caption=source_label, use_container_width=True)

    # Language selection
    lang_name = st.selectbox("🌍 Select Language", list(languages.keys()))
    lang = languages[lang_name]

    if st.button("🚀 Generate Caption"):

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
                confidence_rows.append(f"{label} ({conf:.2f})")

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
            translated = GoogleTranslator(source='auto', target=lang).translate(caption)

            # Voice
            tts = gTTS(translated)
            tts.save("voice.mp3")

        # Display Results
        st.success("✅ Caption Generated")

        st.subheader("📝 Caption:")
        st.write(caption)

        st.subheader("🌍 Translated:")
        st.write(translated)

        st.subheader("🖼️ Detection Preview:")
        st.image(annotated_rgb, caption="YOLO Bounding Boxes", use_container_width=True)

        st.subheader("🏷️ Detected Labels:")
        if detected:
            for label in detected:
                st.write(f"- {label}")
        else:
            st.write("No objects detected")

        st.subheader("🧠 Confidence Scores:")
        if confidence_rows:
            for row in confidence_rows:
                st.write(f"- {row}")
        else:
            st.write("No confidence scores available")

        st.subheader("🔍 Detected Objects:")
        st.write(dict(object_count))

        # Person count
        person_count = object_count.get("person", 0)
        st.subheader("👤 Person Count:")
        st.write(person_count)

        # Audio
        st.subheader("🔊 Voice Output:")
        st.audio("voice.mp3", format="audio/mp3", autoplay=True)

        # Download
        st.download_button(
            label="⬇️ Download Caption",
            data=caption,
            file_name="caption.txt"
        )