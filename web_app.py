"""Serve the supplied interface with real YOLO and BLIP inference."""
import base64
import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent
INFERENCE_LOCK = Lock()
LANGUAGES = {"en", "hi", "te", "ta", "es", "fr", "de", "ja"}


@lru_cache(maxsize=1)
def models():
    from ultralytics import YOLO
    from transformers import BlipProcessor, BlipForConditionalGeneration
    name = "Salesforce/blip-image-captioning-base"
    return (YOLO(str(ROOT / "yolov8m.pt")), BlipProcessor.from_pretrained(name),
            BlipForConditionalGeneration.from_pretrained(name).eval())


@lru_cache(maxsize=256)
def translate(caption, lang):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="en", target=lang).translate(caption)


def analyze(payload):
    from PIL import Image
    import torch
    lang = payload.get("lang", "en")
    if lang not in LANGUAGES:
        raise ValueError("Unsupported language")
    encoded = payload["image"].split(",", 1)[1]
    image = Image.open(BytesIO(base64.b64decode(encoded, validate=True)))
    if image.width * image.height > 20_000_000:
        raise ValueError("Image is too large")
    image = image.convert("RGB")
    with INFERENCE_LOCK:
        yolo, processor, blip = models()
        detection = yolo(image, verbose=False)[0]
        with torch.inference_mode():
            output = blip.generate(**processor(image, return_tensors="pt"), max_new_tokens=60)
        caption = processor.decode(output[0], skip_special_tokens=True)
    objects = []
    colors = ["#2F4BFF", "#0B7F8C", "#C2410C", "#7C3AED"]
    for i, box in enumerate(detection.boxes):
        x1, y1, x2, y2 = box.xyxyn[0].tolist()
        objects.append({"label": detection.names[int(box.cls[0])],
                        "score": round(float(box.conf[0]) * 100),
                        "color": colors[i % len(colors)],
                        "box": [x1 * 100, y1 * 100, (x2-x1) * 100, (y2-y1) * 100]})
    translated, warning, voice_lang = caption, "", lang
    if lang != "en":
        try:
            translated = translate(caption, lang)
        except Exception:
            warning = "Translation is temporarily unavailable. Showing the English caption."
            voice_lang = "en"
    return dict(caption=caption, translated=translated, objects=objects,
                warning=warning, voice_lang=voice_lang)


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.respond(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self.respond(200, {"status": "ok"})
        else:
            self.respond(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/caption":
            self.respond(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 12_000_000:
                self.respond(413, {"error": "Choose a smaller image."})
                return
            payload = json.loads(self.rfile.read(length))
            self.respond(200, analyze(payload))
        except (ValueError, KeyError, IndexError):
            self.respond(400, {"error": "Invalid image or language."})
        except Exception:
            import traceback
            traceback.print_exc()
            self.respond(500, {"error": "Could not analyze the image. Check the server terminal."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8502"))
    print(f"Image Caption Generator: http://localhost:{port}", flush=True)
    ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), port), Handler).serve_forever()
