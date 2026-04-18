"""
Training pipeline for the Ensemble Speech Emotion Recognition model.
Supports stratified split, LR scheduling, checkpointing, and early stopping.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.models.ensemble import EnsembleClassifier
from src.training.dataset import RAVDESSDataset, SEQ_LEN

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def train(data_path: str,
          save_path: str = "models/ensemble_best.pth",
          batch_size: int = 32,
          num_epochs: int = 100,
          lr: float = 1e-3,
          train_split: float = 0.66,
          seed: int = 42,
          patience: int = 15):
    """
    Full training loop.

    Args:
        data_path  : path to RAVDESS root (contains Actor_* folders)
        save_path  : where to save the best model weights
        batch_size : mini-batch size
        num_epochs : maximum training epochs
        lr         : initial learning rate
        train_split: fraction of data used for training
        seed       : random seed for reproducibility
        patience   : early-stopping patience (epochs without val improvement)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    full_dataset = RAVDESSDataset(data_path, seq_len=SEQ_LEN, augment=False)
    indices = list(range(len(full_dataset)))
    labels  = full_dataset.labels

    train_idx, val_idx = train_test_split(
        indices, test_size=1 - train_split,
        stratify=labels, random_state=seed
    )

    train_ds = RAVDESSDataset(data_path, seq_len=SEQ_LEN, augment=True)
    val_ds   = RAVDESSDataset(data_path, seq_len=SEQ_LEN, augment=False)

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"[Trainer] Train: {len(train_idx)} | Val: {len(val_idx)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = EnsembleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float('inf')
    no_improve    = 0
    history       = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for mel, hc, lbl in train_loader:
            mel, hc, lbl = mel.to(device), hc.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(mel, hc), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * mel.size(0)
        train_loss /= len(train_idx)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for mel, hc, lbl in val_loader:
                mel, hc, lbl = mel.to(device), hc.to(device), lbl.to(device)
                logits = model(mel, hc)
                val_loss += criterion(logits, lbl).item() * mel.size(0)
                correct  += (logits.argmax(1) == lbl).sum().item()
                total    += lbl.size(0)
        val_loss /= len(val_idx)
        val_acc   = correct / total

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Trainer] Early stopping at epoch {epoch}")
                break

    print(f"\n[Trainer] Best Val Loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/RAVDESS"
    train(data_path)
