"""
Integration tests for the Flask REST API.
Run with: python -m pytest tests/ -v
"""

import io
import os
import sys
import json
import wave
import struct
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MODEL_PATH'] = 'models/ensemble_best.pth'

from backend.app import app


def make_wav_bytes(duration=2.0, sr=22050, freq=440.0) -> bytes:
    """Generate a minimal valid WAV file in memory."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


# ── Health & Info ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_status_ok(self, client):
        r = client.get('/api/v1/health')
        assert r.status_code == 200

    def test_response_has_status(self, client):
        data = client.get('/api/v1/health').get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'

    def test_response_has_device(self, client):
        data = client.get('/api/v1/health').get_json()
        assert 'device' in data


class TestModelInfoEndpoint:
    def test_status_ok(self, client):
        r = client.get('/api/v1/model-info')
        assert r.status_code == 200

    def test_has_emotions(self, client):
        data = client.get('/api/v1/model-info').get_json()
        assert 'supported_emotions' in data
        assert len(data['supported_emotions']) == 8

    def test_has_params(self, client):
        data = client.get('/api/v1/model-info').get_json()
        assert data['total_parameters'] > 0


# ── Predict Endpoint ──────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_no_file_returns_400(self, client):
        r = client.post('/api/v1/predict')
        assert r.status_code == 400

    def test_valid_wav_returns_200(self, client):
        wav = make_wav_bytes()
        r = client.post('/api/v1/predict',
                        data={'audio': (io.BytesIO(wav), 'test.wav')},
                        content_type='multipart/form-data')
        assert r.status_code == 200

    def test_response_has_emotion(self, client):
        wav = make_wav_bytes()
        data = client.post('/api/v1/predict',
                           data={'audio': (io.BytesIO(wav), 'test.wav')},
                           content_type='multipart/form-data').get_json()
        assert 'emotion' in data
        assert data['emotion'] in [
            'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
        ]

    def test_response_has_confidence(self, client):
        wav = make_wav_bytes()
        data = client.post('/api/v1/predict',
                           data={'audio': (io.BytesIO(wav), 'test.wav')},
                           content_type='multipart/form-data').get_json()
        assert 'confidence' in data
        assert 0.0 <= data['confidence'] <= 1.0

    def test_probabilities_sum_to_one(self, client):
        """Property 2: output probabilities must sum to 1."""
        wav = make_wav_bytes()
        data = client.post('/api/v1/predict',
                           data={'audio': (io.BytesIO(wav), 'test.wav')},
                           content_type='multipart/form-data').get_json()
        probs = list(data['probabilities'].values())
        assert abs(sum(probs) - 1.0) < 1e-4, f"Probs sum = {sum(probs)}"

    def test_probabilities_in_range(self, client):
        wav = make_wav_bytes()
        data = client.post('/api/v1/predict',
                           data={'audio': (io.BytesIO(wav), 'test.wav')},
                           content_type='multipart/form-data').get_json()
        for e, p in data['probabilities'].items():
            assert 0.0 <= p <= 1.0, f"Probability for {e} out of range: {p}"

    def test_unsupported_format_returns_400(self, client):
        r = client.post('/api/v1/predict',
                        data={'audio': (io.BytesIO(b'fake'), 'test.txt')},
                        content_type='multipart/form-data')
        assert r.status_code == 400

    def test_processing_time_present(self, client):
        wav = make_wav_bytes()
        data = client.post('/api/v1/predict',
                           data={'audio': (io.BytesIO(wav), 'test.wav')},
                           content_type='multipart/form-data').get_json()
        assert 'processing_time_ms' in data
        assert data['processing_time_ms'] >= 0


# ── Batch Predict ─────────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_no_files_returns_400(self, client):
        r = client.post('/api/v1/batch-predict')
        assert r.status_code == 400

    def test_multiple_files(self, client):
        wav1 = make_wav_bytes(freq=440)
        wav2 = make_wav_bytes(freq=880)
        r = client.post('/api/v1/batch-predict',
                        data={
                            'audio': [
                                (io.BytesIO(wav1), 'file1.wav'),
                                (io.BytesIO(wav2), 'file2.wav'),
                            ]
                        },
                        content_type='multipart/form-data')
        assert r.status_code == 200
        data = r.get_json()
        assert data['count'] == 2
        assert len(data['predictions']) == 2
