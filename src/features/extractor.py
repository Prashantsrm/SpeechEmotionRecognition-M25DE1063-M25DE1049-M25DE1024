"""
Feature Extraction Pipeline for Speech Emotion Recognition
Extracts MFCC, Mel-Spectrogram, ZCR, RMSE, and Chroma STFT features.
"""

import numpy as np
import librosa
import torch


EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
EMOTION_MAP = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7}

# Feature dimensions
N_MFCC = 13
N_MELS = 64
N_CHROMA = 12
SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512
# Total hand-crafted feature dim per frame: 13 + 1 + 1 + 12 = 27
HAND_CRAFTED_DIM = N_MFCC + 1 + 1 + N_CHROMA  # 27


class FeatureExtractor:
    """Extracts hand-crafted and spectrogram features from raw audio."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr

    # ------------------------------------------------------------------
    # Individual feature extractors
    # ------------------------------------------------------------------

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Return MFCC (n_mfcc, T)."""
        return librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=N_MFCC,
                                    n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Return log-Mel spectrogram (n_mels, T)."""
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=N_MELS,
                                             n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        return librosa.power_to_db(mel, ref=np.max)

    def extract_zcr(self, y: np.ndarray) -> np.ndarray:
        """Return Zero Crossing Rate (1, T)."""
        return librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH,
                                                  hop_length=HOP_LENGTH)

    def extract_rmse(self, y: np.ndarray) -> np.ndarray:
        """Return RMS energy (1, T)."""
        return librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Return Chroma STFT (n_chroma, T)."""
        return librosa.feature.chroma_stft(y=y, sr=self.sr, n_chroma=N_CHROMA,
                                           n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)

    # ------------------------------------------------------------------
    # Combined extractors
    # ------------------------------------------------------------------

    def extract_hand_crafted(self, y: np.ndarray) -> np.ndarray:
        """
        Concatenate MFCC + ZCR + RMSE + Chroma along feature axis.
        Returns array of shape (T, HAND_CRAFTED_DIM).
        """
        mfcc   = self.extract_mfcc(y)        # (13, T)
        zcr    = self.extract_zcr(y)          # (1,  T)
        rmse   = self.extract_rmse(y)         # (1,  T)
        chroma = self.extract_chroma(y)       # (12, T)

        # Align time axis (min T across all features)
        T = min(mfcc.shape[1], zcr.shape[1], rmse.shape[1], chroma.shape[1])
        combined = np.concatenate([mfcc[:, :T], zcr[:, :T],
                                   rmse[:, :T], chroma[:, :T]], axis=0)  # (27, T)
        return combined.T  # (T, 27)

    def extract_mel_3ch(self, y: np.ndarray) -> np.ndarray:
        """
        Return 3-channel mel spectrogram (3, n_mels, T) for CNN input.
        Replicates single channel 3 times to match ImageNet-style input.
        """
        mel = self.extract_mel_spectrogram(y)   # (64, T)
        return np.stack([mel, mel, mel], axis=0)  # (3, 64, T)

    def load_and_extract(self, audio_path: str):
        """
        Load audio file and return (mel_3ch, hand_crafted_features).
        mel_3ch        : np.ndarray (3, 64, T)
        hand_crafted   : np.ndarray (T, 27)
        """
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        # Normalise amplitude
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        mel_3ch = self.extract_mel_3ch(y)
        hand_crafted = self.extract_hand_crafted(y)
        return mel_3ch, hand_crafted


def pad_or_truncate_mel(mel: np.ndarray, target_len: int = 128) -> np.ndarray:
    """Pad or truncate mel spectrogram to fixed time length."""
    C, F, T = mel.shape
    if T >= target_len:
        return mel[:, :, :target_len]
    pad = np.zeros((C, F, target_len - T), dtype=mel.dtype)
    return np.concatenate([mel, pad], axis=2)


def pad_or_truncate_seq(seq: np.ndarray, target_len: int = 128) -> np.ndarray:
    """Pad or truncate sequence to fixed length."""
    T, D = seq.shape
    if T >= target_len:
        return seq[:target_len, :]
    pad = np.zeros((target_len - T, D), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)
