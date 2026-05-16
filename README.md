# AI Image Caption Generator

This project is a simple AI application that helps a user understand an image by combining:

- object detection
- image caption generation
- language translation
- voice output

The app is built with Streamlit and uses YOLO and BLIP models to analyze an uploaded image.

## What This Project Does

When a user uploads an image, the project:

1. detects objects in the image using YOLO
2. generates a natural language caption using BLIP
3. translates the caption into the selected language
4. creates audio for the translated caption
5. shows detected labels, confidence scores, and person count

In short, this project turns an image into text, translated text, and speech.

## How This Project Helps

This project helps users by making image understanding easier and more interactive.

It can be useful for:

- quickly describing uploaded photos
- understanding the main content of an image
- hearing the generated description as audio
- translating image captions into multiple languages
- basic accessibility support for visual content
- demos, student projects, and AI learning

## Why This Project Was Created

The purpose of this project appears to be to show how multiple AI tasks can work together in one application.

Instead of doing only one task, this app combines:

- computer vision for object detection
- caption generation for scene understanding
- translation for multilingual support
- text-to-speech for voice output

This makes it a good mini-project for learning or demonstrating multimodal AI.

## Main Features

- Upload an image from the local system
- Detect objects with bounding boxes
- Generate an automatic image caption
- Translate the caption into:
  - English
  - Hindi
  - Telugu
  - Tamil
  - French
- Play the translated caption as speech
- Show detected object labels and confidence scores
- Count how many people are in the image
- Download the generated caption as a text file

## Project Structure

```text
linkedin/
├── README.md
├── yolov8m.pt
└── i_backend/
    ├── app.py
    └── requirements.txt
```

## Technologies Used

- Python
- Streamlit
- YOLOv8 (`ultralytics`)
- BLIP image captioning model (`transformers`)
- Pillow
- `deep-translator`
- `gTTS`

## How to Run the Project

### 1. Go to the project folder

```bash
cd /Users/sasitamda/Desktop/linkedin
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r i_backend/requirements.txt
```

### 4. Start the Streamlit app

```bash
streamlit run i_backend/app.py
```

### 5. Open the app in your browser

Streamlit will show a local URL, usually:

```text
http://localhost:8501
```

## How to Use the Project

1. Open the app in the browser.
2. Upload a `.jpg`, `.jpeg`, or `.png` image.
3. Choose a language from the dropdown.
4. Click `Generate Caption`.
5. View:
   - generated caption
   - translated caption
   - object detection preview
   - detected labels
   - confidence scores
   - person count
6. Listen to the voice output.
7. Download the generated caption if needed.

## Example Workflow

If a user uploads an image containing two people and a dog, the app may:

- detect `person`, `person`, and `dog`
- generate a caption such as "two people standing with a dog"
- translate that caption into another language
- read the translated caption aloud

## Output Produced by the App

The project can produce:

- an AI-generated caption
- a translated caption
- an annotated image with bounding boxes
- a detected object list
- confidence scores
- person count
- an audio file (`voice.mp3`) during runtime

## Important Notes

- The file `yolov8m.pt` is the YOLO model used by the project.
- On first run, the BLIP model may download from Hugging Face, so internet access may be required.
- The generated file `voice.mp3` is overwritten each time a new caption is generated.
- This project is designed for local use through Streamlit.

## Limitations

- Caption quality depends on the uploaded image and model accuracy.
- Translation quality depends on the translation service.
- Text-to-speech output is generated for one caption at a time.
- The app currently supports image upload only.

## Summary

This project was created to build an AI-powered image understanding tool that can describe an image, translate that description, and speak it aloud. It is useful for demos, learning, accessibility-inspired experiments, and multilingual image captioning tasks.
