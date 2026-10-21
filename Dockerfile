FROM python:3.9-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# librerie Python (aggiunto qdrant-client)
RUN pip install --no-cache-dir \
    essentia-tensorflow \
    fastapi \
    python-multipart \
    uvicorn \
    numpy \
    requests \
    wget \
    qdrant-client

# codice + modelli
COPY server.py .
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
