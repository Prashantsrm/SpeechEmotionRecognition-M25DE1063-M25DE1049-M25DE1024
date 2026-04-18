"""
Evaluation metrics for Speech Emotion Recognition.
Computes accuracy, precision, recall, F1, AUC-ROC, confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def evaluate(model, data_loader, device=None):
    """
    Run inference on data_loader and return a metrics dict.
    Works with both EnsembleClassifier (mel+hc) and plain CNN (mel only).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                mel, hc, lbl = batch
                mel, hc = mel.to(device), hc.to(device)
                logits = model(mel, hc)
            else:
                mel, lbl = batch
                mel = mel.to(device)
                logits = model(mel)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())
            all_probs.extend(probs)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    rep  = classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    except Exception:
        auc = float('nan')

    print(f"\n{'='*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"{'='*55}")
    print(f"\n{rep}")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                auc_roc=auc, confusion_matrix=cm, report=rep,
                all_preds=y_pred, all_labels=y_true, all_probs=y_probs)


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "results/confusion_matrix.png"):
    """Save a confusion-matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix – Ensemble CNN+BiLSTM SER', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Confusion matrix → {save_path}")


def plot_training_curves(history: dict, save_path: str = "results/training_curves.png"):
    """Save loss and accuracy training curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss', color='#7c3aed')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='#ec4899')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['val_acc'], label='Val Accuracy', color='#059669')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Training curves → {save_path}")


def benchmark_inference(model, device=None, n_runs: int = 50):
    """Measure average inference latency in milliseconds."""
    import time
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    mel = torch.randn(1, 3, 64, 128).to(device)
    hc  = torch.randn(1, 128, 27).to(device)

    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            model(mel, hc)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(mel, hc)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    print(f"[Benchmark] Inference latency: {avg_ms:.2f} ± {std_ms:.2f} ms  (n={n_runs}, device={device})")
    return avg_ms
