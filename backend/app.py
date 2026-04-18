"""
Flask REST API for Speech Emotion Recognition.
Uses locally trained CNN + Bi-LSTM Ensemble model (RAVDESS-trained).
Model: models/ensemble_best.pth
Team X - IIT Jodhpur
"""

import io
import os
import sys
import time
import tempfile
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Auto-configure ffmpeg ─────────────────────────────────────────────────────
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = 'ffmpeg'

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
EMOTION_COLORS = {
    'neutral': '#808080', 'calm': '#87CEEB', 'happy': '#FFD700',
    'sad': '#4169E1', 'angry': '#FF4500', 'fearful': '#9932CC',
    'disgust': '#228B22', 'surprised': '#FF69B4',
}
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm', 'opus', 'aac'}
HF_MODEL_ID = "superb/wav2vec2-base-superb-er"

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load Hugging Face model ───────────────────────────────────────────────────
# DISABLED: Using ONLY local trained model
hf_model     = None
hf_processor = None
hf_loaded    = False

print("[API] ⚠️  Hugging Face model DISABLED")
print("[API] Using ONLY local trained ensemble model from RAVDESS")
print("[API] Model: models/ensemble_best.pth")

# ── Local CNN Ensemble Model (PRIMARY - ONLY MODEL) ─────────────────────────────
local_model   = None
local_loaded  = False
MODEL_PATH    = os.environ.get("MODEL_PATH", "models/ensemble_best.pth")

print("[API] Loading local ensemble model...")
try:
    from src.models.ensemble import EnsembleClassifier
    from src.features.extractor import FeatureExtractor, pad_or_truncate_mel, pad_or_truncate_seq
    from src.training.dataset import SEQ_LEN

    local_model = EnsembleClassifier().to(DEVICE)
    extractor   = FeatureExtractor()

    if os.path.exists(MODEL_PATH):
        local_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        local_loaded = True
        print(f"[API] ✅ Local ensemble model loaded from {MODEL_PATH}")
    else:
        local_loaded = False
        print(f"[API] ❌ ERROR: No trained weights found at {MODEL_PATH}")
        print(f"[API] Please train the model first using: python train_ensemble_full.py")
    local_model.eval()
except Exception as e:
    print(f"[API] ❌ ERROR: Failed to load local model: {e}")
    import traceback
    traceback.print_exc()


# ── Audio conversion ──────────────────────────────────────────────────────────

def convert_to_wav(input_path: str, target_sr: int = 16000) -> str:
    """Convert any audio to mono WAV at target_sr using bundled ffmpeg."""
    import soundfile as sf

    # Try librosa first
    try:
        import librosa
        y, _ = librosa.load(input_path, sr=target_sr, mono=True)
        if len(y) > 100:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(tmp.name, y, target_sr)
            return tmp.name
    except Exception:
        pass

    # Fallback: ffmpeg subprocess
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    cmd = [FFMPEG_EXE, '-y', '-i', input_path,
           '-ar', str(target_sr), '-ac', '1', '-f', 'wav', tmp.name]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg conversion failed: {result.stderr.decode()[:300]}")
    return tmp.name


# ── HF prediction ─────────────────────────────────────────────────────────────

def _hf_label_to_ravdess(label: str) -> str:
    """Map HF model label to our RAVDESS emotion names."""
    label = label.lower().strip()
    mapping = {
        # SUPERB model labels
        'ang': 'angry',   'angry': 'angry',
        'hap': 'happy',   'happy': 'happy',   'exc': 'happy',
        'neu': 'neutral', 'neutral': 'neutral',
        'sad': 'sad',     'sadness': 'sad',
        'fru': 'fearful', 'fear': 'fearful',  'fearful': 'fearful',
        'dis': 'disgust', 'disgust': 'disgust',
        'sur': 'surprised','surprised': 'surprised',
        'cal': 'calm',    'calm': 'calm',
        # Generic
        'anger': 'angry', 'joy': 'happy', 'boredom': 'neutral',
        'excited': 'happy', 'ps': 'surprised', 'aw': 'surprised',
    }
    return mapping.get(label, 'neutral')


def predict_hf(wav_path: str) -> dict:
    """Run Hugging Face Wav2Vec2 inference on short audio chunks."""
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # Limit to 5 seconds max to avoid OOM on large files
    MAX_SAMPLES = 16000 * 5
    if len(y) > MAX_SAMPLES:
        # Take middle 5 seconds (more representative than start)
        mid = len(y) // 2
        y = y[mid - MAX_SAMPLES//2 : mid + MAX_SAMPLES//2]

    inputs = hf_processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = hf_model(**inputs).logits
        probs_raw = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Map HF labels → RAVDESS labels and aggregate probabilities
    id2label = hf_model.config.id2label
    ravdess_probs = {e: 0.0 for e in EMOTIONS}

    for idx, prob in enumerate(probs_raw):
        hf_label   = id2label[idx]
        rav_label  = _hf_label_to_ravdess(hf_label)
        ravdess_probs[rav_label] += float(prob)

    # Normalise
    total = sum(ravdess_probs.values())
    if total > 0:
        ravdess_probs = {k: v / total for k, v in ravdess_probs.items()}

    pred_emotion = max(ravdess_probs, key=ravdess_probs.get)
    confidence   = ravdess_probs[pred_emotion]

    return pred_emotion, confidence, ravdess_probs


def predict_local(wav_path: str) -> dict:
    """Run local CNN ensemble inference."""
    import librosa
    from src.features.extractor import pad_or_truncate_mel, pad_or_truncate_seq
    from src.training.dataset import SEQ_LEN

    y, _ = librosa.load(wav_path, sr=22050, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    mel_3ch, hand_crafted = extractor.load_and_extract(wav_path)
    mel_3ch      = pad_or_truncate_mel(mel_3ch, SEQ_LEN)
    hand_crafted = pad_or_truncate_seq(hand_crafted, SEQ_LEN)

    mel_t = torch.FloatTensor(mel_3ch).unsqueeze(0).to(DEVICE)
    hc_t  = torch.FloatTensor(hand_crafted).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs_arr = local_model.predict_proba(mel_t, hc_t)[0].cpu().numpy()

    probs_dict   = {e: round(float(p), 4) for e, p in zip(EMOTIONS, probs_arr)}
    pred_emotion = EMOTIONS[int(np.argmax(probs_arr))]
    confidence   = float(np.max(probs_arr))
    return pred_emotion, confidence, probs_dict


def predict_from_path(audio_path: str) -> dict:
    """Main prediction function — uses ONLY local trained model."""
    t0 = time.time()

    # Convert to WAV
    wav_path = None
    try:
        sr_target = 22050  # Local model uses 22050 Hz
        wav_path  = convert_to_wav(audio_path, target_sr=sr_target)

        if not local_loaded:
            raise RuntimeError("Local model not loaded. Please train the model first.")

        emotion, confidence, probs = predict_local(wav_path)
        model_used = "Local CNN Ensemble (RAVDESS-trained)"

    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "emotion":            emotion,
        "emotion_index":      EMOTIONS.index(emotion),
        "confidence":         round(confidence, 4),
        "color":              EMOTION_COLORS.get(emotion, '#888'),
        "probabilities":      {e: round(float(p), 4) for e, p in probs.items()},
        "processing_time_ms": elapsed_ms,
        "model_used":         model_used,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({
        "status":       "healthy" if local_loaded else "error",
        "local_model":  local_loaded,
        "hf_model":     False,  # Disabled
        "device":       str(DEVICE),
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_in_use": "Local CNN Ensemble (RAVDESS-trained)" if local_loaded else "NONE",
    })


@app.route('/api/v1/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_name":         "Local CNN Ensemble (RAVDESS-trained)",
        "model_type":         "Ensemble (CNN + Bi-LSTM)",
        "training_dataset":   "RAVDESS",
        "hf_model_loaded":    False,  # Disabled
        "local_model_loaded": local_loaded,
        "supported_emotions": EMOTIONS,
        "input_sample_rate":  22050,
        "model_path":         MODEL_PATH,
        "note":               "Using ONLY local trained model. Hugging Face model is disabled.",
    })


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file. Use key 'audio'."}), 400

    audio_file   = request.files['audio']
    filename     = audio_file.filename or 'recording'
    content_type = audio_file.content_type or ''

    # Determine suffix from content-type (browser sends audio/webm)
    if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        if 'webm' in content_type:   suffix = '.webm'
        elif 'ogg' in content_type:  suffix = '.ogg'
        elif 'mp4' in content_type:  suffix = '.mp4'
        elif 'wav' in content_type:  suffix = '.wav'
        else:                         suffix = '.webm'
    else:
        suffix = '.' + filename.rsplit('.', 1)[1].lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_from_path(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/api/v1/batch-predict', methods=['POST'])
def batch_predict():
    files = request.files.getlist('audio')
    if not files:
        return jsonify({"error": "No audio files provided."}), 400

    results = []
    for audio_file in files:
        suffix = '.webm'
        if audio_file.filename and '.' in audio_file.filename:
            suffix = '.' + audio_file.filename.rsplit('.', 1)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        try:
            res = predict_from_path(tmp_path)
            res["filename"] = audio_file.filename
            results.append(res)
        except Exception as e:
            results.append({"filename": audio_file.filename, "error": str(e)})
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return jsonify({"predictions": results, "count": len(results)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[API] Starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
