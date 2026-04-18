"""
RAVDESS Dataset loader compatible with the Ensemble model.
Returns (mel_3ch_tensor, hand_crafted_tensor, label) per sample.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.extractor import (
    FeatureExtractor, pad_or_truncate_mel, pad_or_truncate_seq,
    EMOTION_MAP,
)

SEQ_LEN = 128   # fixed time dimension for padding/truncation


class RAVDESSDataset(Dataset):
    """
    Loads RAVDESS audio files from a directory tree of Actor_* folders.

    Args:
        data_path : root directory containing Actor_* subdirectories
        seq_len   : fixed sequence length for padding/truncation
        augment   : apply simple time-shift augmentation during training
    """

    def __init__(self, data_path: str, seq_len: int = SEQ_LEN, augment: bool = False):
        self.extractor = FeatureExtractor()
        self.seq_len   = seq_len
        self.augment   = augment

        self.file_paths: list[str] = []
        self.labels:     list[int] = []

        actor_dirs = sorted(glob.glob(os.path.join(data_path, "Actor_*")))
        for actor_dir in actor_dirs:
            for wav in glob.glob(os.path.join(actor_dir, "*.wav")):
                parts = os.path.basename(wav).split('-')
                if len(parts) >= 3 and parts[2] in EMOTION_MAP:
                    self.file_paths.append(wav)
                    self.labels.append(EMOTION_MAP[parts[2]])

        print(f"[Dataset] Found {len(self.file_paths)} files in {data_path}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path  = self.file_paths[idx]
        label = self.labels[idx]

        mel_3ch, hand_crafted = self.extractor.load_and_extract(path)

        # Optional time-shift augmentation
        if self.augment:
            shift = np.random.randint(-10, 10)
            mel_3ch    = np.roll(mel_3ch, shift, axis=2)
            hand_crafted = np.roll(hand_crafted, shift, axis=0)

        mel_3ch      = pad_or_truncate_mel(mel_3ch, self.seq_len)      # (3, 64, T)
        hand_crafted = pad_or_truncate_seq(hand_crafted, self.seq_len)  # (T, 27)

        mel_tensor = torch.FloatTensor(mel_3ch)
        hc_tensor  = torch.FloatTensor(hand_crafted)
        lbl_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, hc_tensor, lbl_tensor
