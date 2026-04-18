# Setup and Run Instructions

## Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2: Start Backend API
```bash
python backend/app.py
```

You should see:
```
[API] Loading local ensemble model...
[API] ✅ Local ensemble model loaded from models/ensemble_best.pth
[API] Starting on http://localhost:5000
```

### Step 3: Start Frontend (New Terminal)
```bash
cd frontend
python -m http.server 8000
```

You should see:
```
Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
```

### Step 4: Open in Browser
- **Option A:** Go to `http://localhost:8000`
- **Option B:** Open `frontend/index.html` directly

## System URLs

| Component | URL |
|-----------|-----|
| **Frontend** | http://localhost:8000 |
| **Backend API** | http://localhost:5000 |
| **Health Check** | http://localhost:5000/api/v1/health |
| **Model Info** | http://localhost:5000/api/v1/model-info |

## Testing

### Quick Test
```bash
python test_predict.py
```

Expected output:
```
✅ Model loaded successfully
✅ Prediction made successfully
Emotion: [emotion_name]
Confidence: [0.0-1.0]
```

### Test API Endpoint
```bash
curl http://localhost:5000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "local_model": true,
  "hf_model": false,
  "device": "cpu",
  "model_in_use": "Local CNN Ensemble (RAVDESS-trained)"
}
```

## Model Information

- **Model Type:** Ensemble (CNN + Bi-LSTM)
- **Training Dataset:** RAVDESS
- **Model File:** `models/ensemble_best.pth` (50 MB)
- **Validation Accuracy:** 37.55%
- **Supported Emotions:** 8 classes

## Reproducibility

The system is fully reproducible:

1. **Same Model:** Uses pre-trained weights from `models/ensemble_best.pth`
2. **Same Features:** Feature extraction is deterministic
3. **Same Results:** Predictions are deterministic (no randomness in inference)

When you run the system, you will get the same predictions for the same audio inputs.

## Troubleshooting

### Port 5000 Already in Use
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Port 8000 Already in Use
```bash
# Use different port
python -m http.server 8001
```

### Model Not Found
```
Error: No trained weights found at models/ensemble_best.pth
```
Ensure `models/ensemble_best.pth` exists in the project root.

### Audio Processing Error
Install ffmpeg:
```bash
pip install imageio-ffmpeg
```

## File Structure

```
.
├── backend/
│   ├── app.py                    # Flask API
│   └── requirements.txt          # Dependencies
├── frontend/
│   └── index.html               # Web UI
├── src/
│   ├── features/extractor.py    # Feature extraction
│   ├── models/                  # Model architectures
│   ├── training/                # Training code
│   └── evaluation/              # Evaluation code
├── models/
│   └── ensemble_best.pth        # Trained model
├── results/
│   ├── training_curves.png
│   ├── ensemble_confusion_matrix.png
│   └── training_metrics.json
├── datasets/
│   ├── RAVDESS/                 # Training data
│   └── CREMA-D/                 # Test data
├── train_ensemble_full.py       # Training script
├── evaluate_ravdess_test.py     # Evaluation script
├── test_predict.py              # Quick test
├── README.md                    # Documentation
└── Team_X_Project_Report.pdf    # Final report
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Start backend API
3. ✅ Start frontend server
4. ✅ Open browser and test
5. ✅ Upload audio files or record from microphone
6. ✅ View emotion predictions

---

**Status:** ✅ Ready to Run
**Last Updated:** April 18, 2026
