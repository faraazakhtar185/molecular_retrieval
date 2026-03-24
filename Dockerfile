FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxext6 \
    libxrender1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY demo/requirements.txt demo/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r demo/requirements.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn demo.app:app --host 0.0.0.0 --port ${PORT}"]
