# Speech Emotion Recognition – Enhanced Ensemble System

**Team X - IIT Jodhpur**
- Asit Jain (M25DE1049)
- Avinash Singh (M25DE1024)
- Prashant Kumar Mishra (M25DE1063)

## Overview

This project implements an ensemble-based speech emotion recognition system using the RAVDESS dataset. The system combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (Bi-LSTM) networks to classify emotions from speech signals.

### Key Features
- **Ensemble Architecture**: CNN + Bi-LSTM for robust emotion classification
- **Multi-Feature Extraction**: MFCC, Mel-Spectrogram, ZCR, RMSE, Chroma features
- **REST API Backend**: Flask-based API for predictions
- **Web UI Frontend**: Real-time emotion prediction from microphone or file upload
- **Cross-Dataset Evaluation**: Tested on TESS and CREMA-D datasets

## Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 75.10% |
| **Test Accuracy** | 89.67% |
| **Test F1-Score** | 0.8955 |
| **Test AUC-ROC** | 0.9913 |
| **Cross-Dataset (CREMA-D)** | 16.34% |
| **Training Time** | 34m 39s (53 epochs) |
| **Model Size** | 50 MB |
| **Inference Latency** | 5.85 ms |

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Verify model exists
ls models/ensemble_best.pth
```

## Usage

### 1. Start Backend API

```bash
python backend/app.py
```

The API will start on `http://localhost:5000`

**Health Check:**
```bash
curl http://localhost:5000/api/v1/health
```

### 2. Start Frontend

**Option A: Using Python HTTP Server**
```bash
cd frontend
python -m http.server 8000
```

**Option B: Direct File Access**
- Open `frontend/index.html` in your browser

### 3. Access the System

Open your browser and go to:
- **Frontend**: `http://localhost:8000` (if using HTTP server)
- **Or**: Open `frontend/index.html` directly

### 4. Make Predictions

**Via Web UI:**
1. Record audio using microphone or upload WAV file
2. Click "Predict Emotion"
3. View results with confidence scores

**Via API:**
```bash
curl -X POST -F "audio=@sample.wav" http://localhost:5000/api/v1/predict
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```
Returns system status and model information.

### Model Info
```
GET /api/v1/model-info
```
Returns model details and supported emotions.

### Single Prediction
```
POST /api/v1/predict
```
**Input:** Audio file (WAV, MP3, OGG, FLAC, M4A, WebM, Opus, AAC)
**Output:** Emotion prediction with confidence scores

**Example:**
```bash
curl -X POST -F "audio=@audio.wav" http://localhost:5000/api/v1/predict
```

### Batch Prediction
```
POST /api/v1/batch-predict
```
**Input:** Multiple audio files
**Output:** Array of predictions

## Project Structure

```
.
├── backend/
│   ├── app.py                 # Flask REST API
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── index.html            # Web UI
├── src/
│   ├── features/
│   │   └── extractor.py      # Feature extraction
│   ├── models/
│   │   ├── cnn_branch.py     # CNN architecture
│   │   ├── lstm_branch.py    # Bi-LSTM architecture
│   │   └── ensemble.py       # Ensemble classifier
│   ├── training/
│   │   ├── dataset.py        # Dataset loader
│   │   └── trainer.py        # Training pipeline
│   └── evaluation/
│       └── metrics.py        # Evaluation metrics
├── models/
│   └── ensemble_best.pth     # Trained model weights
├── results/
│   ├── training_curves.png
│   ├── ensemble_confusion_matrix.png
│   ├── tess_confusion_matrix.png
│   └── training_metrics.json
├── datasets/
│   ├── RAVDESS/              # Training dataset
│   └── CREMA-D/              # Cross-dataset evaluation
├── train_ensemble_full.py    # Training script
├── evaluate_ravdess_test.py  # Test evaluation
├── evaluate_cross_dataset.py # Cross-dataset validation
├── test_predict.py           # Quick prediction test
└── README.md                 # This file
```

## Training

To retrain the model:

```bash
python train_ensemble_full.py --data path/to/RAVDESS --epochs 100
```

**Note:** The RAVDESS dataset must be downloaded separately and placed in `datasets/RAVDESS/`

## Evaluation

### Test Set Evaluation
```bash
python evaluate_ravdess_test.py --data path/to/RAVDESS --model models/ensemble_best.pth
```

### Cross-Dataset Evaluation
```bash
python evaluate_cross_dataset.py --model models/ensemble_best.pth
```

## Model Architecture

### CNN Branch
- Processes Mel-Spectrogram features
- Extracts spatial patterns from spectrograms
- 3 convolutional layers with max pooling

### Bi-LSTM Branch
- Processes MFCC and hand-crafted features
- Captures temporal dependencies
- 2 bidirectional LSTM layers

### Ensemble
- Concatenates CNN and Bi-LSTM outputs
- Fully connected layers for classification
- 8-class emotion output

## Supported Emotions

1. **Neutral** - No emotion
2. **Calm** - Peaceful, relaxed
3. **Happy** - Joyful, cheerful
4. **Sad** - Sorrowful, melancholic
5. **Angry** - Furious, irritated
6. **Fearful** - Scared, anxious
7. **Disgust** - Repulsed, contemptuous
8. **Surprised** - Astonished, amazed

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Weight Decay | 0.0001 |
| Dropout | 0.5 |
| Scheduler | ReduceLROnPlateau |
| Train-Validation Split | 66%-34% |
| Epochs Trained | 53 (early stopping, patience=15) |

## Dataset Information

### RAVDESS (Training)
- **Actors:** 24 professional actors
- **Emotions:** 8 classes
- **Samples:** 1,440 (training) + 300 (test)
- **Format:** WAV, 16 kHz, mono
- **Source:** [RAVDESS Dataset](https://zenodo.org/record/1188976)

### TESS (Cross-Dataset Evaluation)
- **Speakers:** 2 female speakers
- **Emotions:** 7 classes
- **Samples:** 2,800
- **Format:** WAV, 16 kHz, mono

### CREMA-D (Cross-Dataset Evaluation)
- **Actors:** 91 actors
- **Emotions:** 6 classes
- **Samples:** 7,442
- **Format:** WAV, 16 kHz, mono

## System Requirements

- **CPU:** Intel i5 or equivalent
- **RAM:** 8 GB minimum
- **Storage:** 10 GB (including datasets)
- **GPU:** Optional (CUDA-compatible for faster training)

## Troubleshooting

### Model Not Loading
```
Error: No trained weights found at models/ensemble_best.pth
```
**Solution:** Ensure the model file exists. Train using `python train_ensemble_full.py`

### Audio Processing Error
```
Error: ffmpeg conversion failed
```
**Solution:** Install ffmpeg:
- **Windows:** `pip install imageio-ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`
- **macOS:** `brew install ffmpeg`

### Port Already in Use
```
Error: Address already in use
```
**Solution:** Change port in `backend/app.py` or kill existing process:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :5000
kill -9 <PID>
```

## References

1. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Emotion Database (RAVDESS): A database of actors performing scripted emotional expressions. PLoS ONE, 13(5), e0196424.

2. Dupont, S., & Luettin, J. (2000). Audio-visual speech modeling for continuous speech recognition. IEEE Transactions on Multimedia, 2(3), 141-151.

3. Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5-6), 602-610.

4. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25.

## License

This project is for educational purposes.

## Contact

For questions or issues, please contact the team members.

---

**Last Updated:** April 18, 2026
**Status:** ✅ Ready for Deployment
