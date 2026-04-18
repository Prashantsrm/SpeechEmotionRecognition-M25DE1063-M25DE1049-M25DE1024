"""
Bi-directional LSTM Branch.
Processes sequences of hand-crafted features.
Output: 256-dimensional feature vector (128 forward + 128 backward).
"""

import torch
import torch.nn as nn


class BiLSTMBranch(nn.Module):
    """
    2-layer Bidirectional LSTM for temporal feature processing.
    Input : (B, T, input_size)
    Output: (B, 2 * hidden_size)  →  (B, 256)
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * 2  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size)
        Returns:
            (B, 256)
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, hidden_size)
        # Take last layer's forward and backward hidden states
        forward_h  = h_n[-2]   # (B, 128)
        backward_h = h_n[-1]   # (B, 128)
        out = torch.cat([forward_h, backward_h], dim=1)  # (B, 256)
        return self.dropout(out)
