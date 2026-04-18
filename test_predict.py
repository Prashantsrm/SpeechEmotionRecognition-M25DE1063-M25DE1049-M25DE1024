"""
Test end-to-end prediction on a real audio file.
Usage: python test_predict.py path/to/audio.wav
"""
import sys
import os
import glob

# Find a wav file to test with
if len(sys.argv) > 1:
    audio_path = sys.argv[1]
else:
    wavs = glob.glob("**/*.wav", recursive=True)
    if not wavs:
        print("No WAV files found. Provide path as argument.")
        sys.exit(1)
    audio_path = wavs[0]

print(f"Testing with: {audio_path}")

from src.features.extractor import FeatureExtractor, pad_or_truncate_mel, pad_or_truncate_seq
from src.models.ensemble import EnsembleClassifier
from src.training.dataset import SEQ_LEN
import torch
import numpy as np

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

extractor = FeatureExtractor()
model = EnsembleClassifier()
model.eval()

mel_3ch, hand_crafted = extractor.load_and_extract(audio_path)
mel_3ch      = pad_or_truncate_mel(mel_3ch, SEQ_LEN)
hand_crafted = pad_or_truncate_seq(hand_crafted, SEQ_LEN)

mel_t = torch.FloatTensor(mel_3ch).unsqueeze(0)
hc_t  = torch.FloatTensor(hand_crafted).unsqueeze(0)

with torch.no_grad():
    probs = model.predict_proba(mel_t, hc_t)[0].numpy()

pred_idx = int(np.argmax(probs))
print(f"\nPredicted emotion : {EMOTIONS[pred_idx]}")
print(f"Confidence        : {probs[pred_idx]:.4f} ({probs[pred_idx]*100:.1f}%)")
print("\nAll probabilities:")
for e, p in sorted(zip(EMOTIONS, probs), key=lambda x: -x[1]):
    bar = '█' * int(p * 30)
    print(f"  {e:12s} {p:.4f}  {bar}")

print("\n✅ Prediction pipeline working correctly!")
