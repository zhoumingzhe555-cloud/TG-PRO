FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    libgomp1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data/images /data/models /data/imports

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<3" \
 && pip install -r requirements.txt
COPY . .

CMD ["python", "start.py"]
