"""
Test emotion prediction on the two sample audio files.
Saves results to: audio_test_results.txt

Audio files tested:
  1. alban_gogh-quot-panic-fear-quot-sound-effect-479998.mp3
  2. Piya Aaye Na(KoshalWorld.Com).mp3

Team X - IIT Jodhpur
"""

import os
import sys
import torch
import numpy as np
import json
import tempfile
import subprocess
from datetime import datetime

# ── Setup ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
MODEL_PATH = "models/ensemble_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_FILES = [
    "alban_gogh-quot-panic-fear-quot-sound-effect-479998.mp3",
    "Piya Aaye Na(KoshalWorld.Com).mp3"
]

AUDIO_DESCRIPTIONS = {
    "alban_gogh-quot-panic-fear-quot-sound-effect-479998.mp3": "Panic/Fear Sound Effect",
    "Piya Aaye Na(KoshalWorld.Com).mp3": "Hindi Song (Piya Aaye Na)"
}


def convert_to_wav(input_path, target_sr=22050):
    """Convert audio file to WAV format using librosa or ffmpeg."""
    import soundfile as sf

    # Try librosa first
    try:
        import librosa
        y, _ = librosa.load(input_path, sr=target_sr, mono=True)
        if len(y) > 100:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(tmp.name, y, target_sr)
            return tmp.name
    except Exception as e:
        print(f"  [librosa failed: {e}, trying ffmpeg...]")

    # Fallback: ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = 'ffmpeg'

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    cmd = [ffmpeg_exe, '-y', '-i', input_path,
           '-ar', str(target_sr), '-ac', '1', '-f', 'wav', tmp.name]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    return tmp.name


def load_model():
    """Load the trained ensemble model."""
    from src.models.ensemble import EnsembleClassifier
    model = EnsembleClassifier().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"  ✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"  ⚠️  No trained weights found. Using random weights.")
    model.eval()
    return model


def predict(model, audio_path):
    """Run prediction on an audio file."""
    from src.features.extractor import FeatureExtractor, pad_or_truncate_mel, pad_or_truncate_seq
    from src.training.dataset import SEQ_LEN

    extractor = FeatureExtractor()

    # Convert to WAV
    wav_path = convert_to_wav(audio_path, target_sr=22050)

    try:
        mel_3ch, hand_crafted = extractor.load_and_extract(wav_path)
        mel_3ch      = pad_or_truncate_mel(mel_3ch, SEQ_LEN)
        hand_crafted = pad_or_truncate_seq(hand_crafted, SEQ_LEN)

        mel_t = torch.FloatTensor(mel_3ch).unsqueeze(0).to(DEVICE)
        hc_t  = torch.FloatTensor(hand_crafted).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs_arr = model.predict_proba(mel_t, hc_t)[0].cpu().numpy()

        probs_dict   = {e: round(float(p), 4) for e, p in zip(EMOTIONS, probs_arr)}
        pred_emotion = EMOTIONS[int(np.argmax(probs_arr))]
        confidence   = float(np.max(probs_arr))

        return pred_emotion, confidence, probs_dict

    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def bar_chart(value, width=30):
    """Create a simple text bar chart."""
    filled = int(value * width)
    return '█' * filled + '░' * (width - filled)


def main():
    print("=" * 70)
    print("  SPEECH EMOTION RECOGNITION - AUDIO SAMPLE TEST")
    print("  Team X | IIT Jodhpur")
    print(f"  Date: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
    print("=" * 70)
    print()

    # Load model
    print("Loading model...")
    model = load_model()
    print()

    results = []

    for audio_file in AUDIO_FILES:
        print("-" * 70)
        desc = AUDIO_DESCRIPTIONS.get(audio_file, audio_file)
        print(f"  File       : {audio_file}")
        print(f"  Description: {desc}")
        print()

        if not os.path.exists(audio_file):
            print(f"  ❌ File not found: {audio_file}")
            results.append({
                "file": audio_file,
                "description": desc,
                "error": "File not found"
            })
            continue

        try:
            print("  Processing audio...")
            emotion, confidence, probs = predict(model, audio_file)

            print(f"  🎯 Predicted Emotion : {emotion.upper()}")
            print(f"  📊 Confidence        : {confidence:.4f} ({confidence*100:.1f}%)")
            print()
            print("  All Emotion Probabilities:")
            print("  " + "-" * 50)

            sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
            for emo, prob in sorted_probs:
                marker = " ◄ PREDICTED" if emo == emotion else ""
                print(f"  {emo:12s}  {prob:.4f}  {bar_chart(prob)}{marker}")

            results.append({
                "file": audio_file,
                "description": desc,
                "predicted_emotion": emotion,
                "confidence": round(confidence, 4),
                "confidence_percent": f"{confidence*100:.1f}%",
                "all_probabilities": probs
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "file": audio_file,
                "description": desc,
                "error": str(e)
            })

        print()

    # ── Save results ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    # Save as TXT
    txt_path = "audio_test_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  SPEECH EMOTION RECOGNITION - AUDIO SAMPLE TEST RESULTS\n")
        f.write("  Team X | IIT Jodhpur\n")
        f.write(f"  Date: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}\n")
        f.write("  Model: Local CNN Ensemble (RAVDESS-trained)\n")
        f.write("  Model Path: models/ensemble_best.pth\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            f.write("-" * 70 + "\n")
            f.write(f"  File       : {r['file']}\n")
            f.write(f"  Description: {r['description']}\n")
            if "error" in r:
                f.write(f"  ERROR      : {r['error']}\n")
            else:
                f.write(f"\n")
                f.write(f"  Predicted Emotion : {r['predicted_emotion'].upper()}\n")
                f.write(f"  Confidence        : {r['confidence']} ({r['confidence_percent']})\n")
                f.write(f"\n")
                f.write(f"  All Emotion Probabilities:\n")
                f.write("  " + "-" * 50 + "\n")
                sorted_probs = sorted(r['all_probabilities'].items(), key=lambda x: -x[1])
                for emo, prob in sorted_probs:
                    marker = " <-- PREDICTED" if emo == r['predicted_emotion'] else ""
                    f.write(f"  {emo:12s}  {prob:.4f}  {bar_chart(prob)}{marker}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("  SUMMARY\n")
        f.write("=" * 70 + "\n")
        for r in results:
            if "error" not in r:
                f.write(f"  {r['description']:<40} -> {r['predicted_emotion'].upper():<12} ({r['confidence_percent']})\n")
        f.write("\n")
        f.write("  Note: Model trained on RAVDESS dataset (speech only).\n")
        f.write("  Music files may not predict accurately as model is trained on speech.\n")
        f.write("=" * 70 + "\n")

    print(f"  ✅ Results saved to: {txt_path}")

    # Save as JSON
    json_path = "audio_test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "model": "Local CNN Ensemble (RAVDESS-trained)",
            "model_path": MODEL_PATH,
            "results": results
        }, f, indent=2)

    print(f"  ✅ Results saved to: {json_path}")
    print()

    # Final summary
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" not in r:
            print(f"  {r['description']:<40} -> {r['predicted_emotion'].upper():<12} ({r['confidence_percent']})")
        else:
            print(f"  {r['description']:<40} -> ERROR: {r['error']}")
    print()
    print("  Note: Model is trained on RAVDESS (acted speech).")
    print("  Music/non-speech audio may give unexpected results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
