"""
CNN Branch with Depthwise Separable Convolutions.
Extracts spatial features from mel-spectrograms.
Output: 512-dimensional feature vector.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv followed by pointwise conv + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size,
                            stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pw(self.dw(x))))


class CNNBranch(nn.Module):
    """
    Lightweight CNN for mel-spectrogram feature extraction.
    Input : (B, 3, 64, T)
    Output: (B, 512)
    """

    def __init__(self, input_channels: int = 3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Depthwise separable blocks
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)   # (B, 512)
        return self.dropout(x)
