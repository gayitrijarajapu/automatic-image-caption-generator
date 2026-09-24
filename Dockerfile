FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user web_app.py index.html yolov8m.pt ./
USER user
ENV HF_HOME=/home/user/.cache/huggingface HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "web_app.py"]
