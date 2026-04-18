FROM python:3.11-slim

WORKDIR /app

# System deps for librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/       ./src/
COPY backend/   ./backend/
COPY run_api.py .

# Create models dir (mount or copy your trained .pth here)
RUN mkdir -p models results

ENV MODEL_PATH=models/ensemble_best.pth
ENV PORT=5000

EXPOSE 5000

CMD ["python", "run_api.py"]
