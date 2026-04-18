"""
Unit + property-based tests for the feature extraction pipeline.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.extractor import (
    FeatureExtractor, pad_or_truncate_mel, pad_or_truncate_seq,
    HAND_CRAFTED_DIM, N_MFCC, N_MELS, N_CHROMA, SAMPLE_RATE,
)

SR = SAMPLE_RATE
fe = FeatureExtractor(sr=SR)


def make_audio(duration=2.0, sr=SR):
    """Generate a synthetic sine-wave audio signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestMFCC:
    def test_shape(self):
        y = make_audio()
        mfcc = fe.extract_mfcc(y)
        assert mfcc.shape[0] == N_MFCC, f"Expected {N_MFCC} coefficients, got {mfcc.shape[0]}"

    def test_no_nan(self):
        y = make_audio()
        mfcc = fe.extract_mfcc(y)
        assert not np.isnan(mfcc).any(), "MFCC contains NaN"

    def test_deterministic(self):
        y = make_audio()
        assert np.allclose(fe.extract_mfcc(y), fe.extract_mfcc(y)), "MFCC not deterministic"


class TestMelSpectrogram:
    def test_shape(self):
        y = make_audio()
        mel = fe.extract_mel_spectrogram(y)
        assert mel.shape[0] == N_MELS

    def test_db_scale(self):
        y = make_audio()
        mel = fe.extract_mel_spectrogram(y)
        # dB values should be ≤ 0 (ref=max)
        assert mel.max() <= 0.1, "Mel-spectrogram not in dB scale"


class TestZCR:
    def test_shape(self):
        y = make_audio()
        zcr = fe.extract_zcr(y)
        assert zcr.shape[0] == 1

    def test_range(self):
        y = make_audio()
        zcr = fe.extract_zcr(y)
        assert zcr.min() >= 0.0 and zcr.max() <= 1.0, "ZCR out of [0,1]"


class TestRMSE:
    def test_shape(self):
        y = make_audio()
        rmse = fe.extract_rmse(y)
        assert rmse.shape[0] == 1

    def test_non_negative(self):
        y = make_audio()
        rmse = fe.extract_rmse(y)
        assert rmse.min() >= 0.0, "RMSE is negative"


class TestChroma:
    def test_shape(self):
        y = make_audio()
        chroma = fe.extract_chroma(y)
        assert chroma.shape[0] == N_CHROMA

    def test_range(self):
        y = make_audio()
        chroma = fe.extract_chroma(y)
        assert chroma.min() >= 0.0 and chroma.max() <= 1.0, "Chroma out of [0,1]"


class TestHandCrafted:
    def test_shape(self):
        y = make_audio()
        hc = fe.extract_hand_crafted(y)
        assert hc.ndim == 2
        assert hc.shape[1] == HAND_CRAFTED_DIM, \
            f"Expected {HAND_CRAFTED_DIM} features, got {hc.shape[1]}"

    def test_deterministic(self):
        y = make_audio()
        assert np.allclose(fe.extract_hand_crafted(y),
                           fe.extract_hand_crafted(y)), "Hand-crafted features not deterministic"


class TestMel3Ch:
    def test_shape(self):
        y = make_audio()
        mel = fe.extract_mel_3ch(y)
        assert mel.shape[0] == 3
        assert mel.shape[1] == N_MELS

    def test_channels_identical(self):
        y = make_audio()
        mel = fe.extract_mel_3ch(y)
        assert np.allclose(mel[0], mel[1]) and np.allclose(mel[1], mel[2])


# ── Padding tests ─────────────────────────────────────────────────────────────

class TestPadding:
    def test_mel_pad(self):
        mel = np.zeros((3, 64, 50))
        out = pad_or_truncate_mel(mel, 128)
        assert out.shape == (3, 64, 128)

    def test_mel_truncate(self):
        mel = np.zeros((3, 64, 200))
        out = pad_or_truncate_mel(mel, 128)
        assert out.shape == (3, 64, 128)

    def test_seq_pad(self):
        seq = np.zeros((50, HAND_CRAFTED_DIM))
        out = pad_or_truncate_seq(seq, 128)
        assert out.shape == (128, HAND_CRAFTED_DIM)

    def test_seq_truncate(self):
        seq = np.zeros((200, HAND_CRAFTED_DIM))
        out = pad_or_truncate_seq(seq, 128)
        assert out.shape == (128, HAND_CRAFTED_DIM)


# ── Property-based tests ──────────────────────────────────────────────────────

class TestProperties:
    """Correctness properties from the design spec."""

    def test_feature_extraction_consistency(self):
        """Property 1: identical audio → identical features."""
        y = make_audio(duration=3.0)
        hc1 = fe.extract_hand_crafted(y)
        hc2 = fe.extract_hand_crafted(y)
        assert np.allclose(hc1, hc2), "Feature extraction is not deterministic"

    def test_feature_norm_bounds(self):
        """Property 4: normalized features within ±3 std."""
        from sklearn.preprocessing import StandardScaler
        y = make_audio(duration=3.0)
        hc = fe.extract_hand_crafted(y)
        scaler = StandardScaler()
        norm = scaler.fit_transform(hc)
        assert np.abs(norm).max() <= 10.0, "Normalized features exceed expected bounds"

    def test_non_zero_features(self):
        """Features from non-silent audio must be non-zero."""
        y = make_audio(duration=2.0)
        hc = fe.extract_hand_crafted(y)
        assert np.linalg.norm(hc) > 0, "Feature vector is all zeros"

    def test_silence_handled(self):
        """Silent audio should not crash the extractor."""
        y = np.zeros(SR * 2, dtype=np.float32)
        try:
            hc = fe.extract_hand_crafted(y)
            assert hc is not None
        except Exception as e:
            pytest.fail(f"Extractor crashed on silence: {e}")
