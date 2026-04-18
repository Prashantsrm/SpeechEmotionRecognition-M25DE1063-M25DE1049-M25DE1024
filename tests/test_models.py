"""
Unit tests for CNN, LSTM, and Ensemble model components.
Run with: python -m pytest tests/ -v
"""

import torch
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_branch import CNNBranch
from src.models.lstm_branch import BiLSTMBranch
from src.models.ensemble import EnsembleClassifier
from src.features.extractor import HAND_CRAFTED_DIM

B   = 4      # batch size
T   = 128    # sequence length
DIM = HAND_CRAFTED_DIM  # 27


# ── CNN Branch ────────────────────────────────────────────────────────────────

class TestCNNBranch:
    def setup_method(self):
        self.model = CNNBranch(input_channels=3)
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(B, 3, 64, T)
        out = self.model(x)
        assert out.shape == (B, 512), f"Expected (B,512), got {out.shape}"

    def test_no_nan(self):
        x = torch.randn(B, 3, 64, T)
        out = self.model(x)
        assert not torch.isnan(out).any()

    def test_batch_size_1(self):
        x = torch.randn(1, 3, 64, T)
        out = self.model(x)
        assert out.shape == (1, 512)

    def test_different_time_lengths(self):
        for t in [64, 128, 256]:
            x = torch.randn(2, 3, 64, t)
            out = self.model(x)
            assert out.shape == (2, 512), f"Failed for T={t}"


# ── Bi-LSTM Branch ────────────────────────────────────────────────────────────

class TestBiLSTMBranch:
    def setup_method(self):
        self.model = BiLSTMBranch(input_size=DIM, hidden_size=128)
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(B, T, DIM)
        out = self.model(x)
        assert out.shape == (B, 256), f"Expected (B,256), got {out.shape}"

    def test_output_dim_attribute(self):
        assert self.model.output_dim == 256

    def test_no_nan(self):
        x = torch.randn(B, T, DIM)
        out = self.model(x)
        assert not torch.isnan(out).any()

    def test_batch_size_1(self):
        x = torch.randn(1, T, DIM)
        out = self.model(x)
        assert out.shape == (1, 256)

    def test_different_seq_lengths(self):
        for t in [32, 64, 128, 256]:
            x = torch.randn(2, t, DIM)
            out = self.model(x)
            assert out.shape == (2, 256), f"Failed for T={t}"


# ── Ensemble Classifier ───────────────────────────────────────────────────────

class TestEnsembleClassifier:
    def setup_method(self):
        self.model = EnsembleClassifier(hand_crafted_dim=DIM)
        self.model.eval()

    def _inputs(self, b=B):
        mel = torch.randn(b, 3, 64, T)
        hc  = torch.randn(b, T, DIM)
        return mel, hc

    def test_logits_shape(self):
        mel, hc = self._inputs()
        logits = self.model(mel, hc)
        assert logits.shape == (B, 8), f"Expected (B,8), got {logits.shape}"

    def test_proba_sums_to_one(self):
        """Property 2: output is valid probability distribution."""
        mel, hc = self._inputs()
        probs = self.model.predict_proba(mel, hc)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"Probabilities don't sum to 1: {sums}"

    def test_proba_in_range(self):
        """All probabilities must be in [0, 1]."""
        mel, hc = self._inputs()
        probs = self.model.predict_proba(mel, hc)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_argmax_valid(self):
        """Predicted class index must be in [0, 7]."""
        mel, hc = self._inputs()
        probs = self.model.predict_proba(mel, hc)
        preds = probs.argmax(dim=1)
        assert preds.min() >= 0 and preds.max() <= 7

    def test_no_nan(self):
        mel, hc = self._inputs()
        logits = self.model(mel, hc)
        assert not torch.isnan(logits).any()

    def test_determinism(self):
        """Property 3: same input → same output (eval mode, no dropout)."""
        mel, hc = self._inputs()
        with torch.no_grad():
            out1 = self.model(mel, hc)
            out2 = self.model(mel, hc)
        assert torch.allclose(out1, out2), "Model is not deterministic in eval mode"

    def test_batch_size_1(self):
        mel, hc = self._inputs(b=1)
        logits = self.model(mel, hc)
        assert logits.shape == (1, 8)

    def test_parameter_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total < 5_000_000, f"Model too large: {total:,} params"
        print(f"\n  Total parameters: {total:,}")

    def test_fusion_preserves_info(self):
        """Property 5: fused features are non-zero."""
        mel, hc = self._inputs()
        cnn_feat  = self.model.cnn_branch(mel)
        lstm_feat = self.model.lstm_branch(hc)
        fused = torch.cat([cnn_feat, lstm_feat], dim=1)
        assert fused.shape[1] == 768, f"Expected 768, got {fused.shape[1]}"
        assert fused.norm() > 0, "Fused features are all zeros"
