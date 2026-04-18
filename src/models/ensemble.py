"""
Ensemble Classifier: CNN + Bi-LSTM fusion for Speech Emotion Recognition.
CNN branch  → 512-dim spatial features from mel-spectrogram
LSTM branch → 256-dim temporal features from hand-crafted features
Fusion      → 768 → 256 → 128 → 8 (emotions)
"""

import torch
import torch.nn as nn

from src.models.cnn_branch import CNNBranch
from src.models.lstm_branch import BiLSTMBranch
from src.features.extractor import HAND_CRAFTED_DIM


class EnsembleClassifier(nn.Module):
    """
    Ensemble model combining CNN and Bi-LSTM branches.

    Inputs:
        mel_spec          : (B, 3, 64, T)   – 3-channel mel spectrogram
        hand_crafted_feats: (B, T, 27)       – hand-crafted feature sequence

    Output:
        logits            : (B, 8)           – raw class scores (use softmax for probs)
    """

    CNN_OUT  = 512
    LSTM_OUT = 256
    NUM_CLASSES = 8

    def __init__(self, hand_crafted_dim: int = HAND_CRAFTED_DIM):
        super().__init__()

        self.cnn_branch  = CNNBranch(input_channels=3)
        self.lstm_branch = BiLSTMBranch(input_size=hand_crafted_dim, hidden_size=128)

        fusion_in = self.CNN_OUT + self.LSTM_OUT  # 768

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.NUM_CLASSES),
        )

    def forward(self, mel_spec: torch.Tensor,
                hand_crafted: torch.Tensor) -> torch.Tensor:
        cnn_feat  = self.cnn_branch(mel_spec)       # (B, 512)
        lstm_feat = self.lstm_branch(hand_crafted)  # (B, 256)

        fused  = torch.cat([cnn_feat, lstm_feat], dim=1)  # (B, 768)
        fused  = self.fusion(fused)                        # (B, 256)
        logits = self.classifier(fused)                    # (B, 8)
        return logits

    def predict_proba(self, mel_spec: torch.Tensor,
                      hand_crafted: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return torch.softmax(self.forward(mel_spec, hand_crafted), dim=1)
