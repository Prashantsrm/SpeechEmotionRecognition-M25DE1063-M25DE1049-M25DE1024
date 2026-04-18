# Implementation Guide: Step-by-Step Code Examples
## Enhanced Speech Emotion Recognition System

---

## Quick Start (5 Minutes)

### 1. Load Existing Model

```python
import torch
import numpy as np
import librosa

# Load existing PyTorch model
model = torch.load('emotion_model_group8.pth')
model.eval()  # Set to evaluation mode

# Or load Keras model
from tensorflow.keras.models import load_model
model_keras = load_model('Emotions_Model.h5')
```

### 2. Load Audio File

```python
# Load audio file
audio_path = 'sample_audio.wav'
y, sr = librosa.load(audio_path, sr=22050)

print(f"Audio shape: {y.shape}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(y) / sr:.2f} seconds")
```

### 3. Extract Features

```python
# Extract MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(f"MFCC shape: {mfcc.shape}")  # (13, time_steps)

# Extract Mel-Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
print(f"Mel-Spectrogram shape: {mel_spec_db.shape}")  # (64, time_steps)
```

### 4. Make Prediction

```python
# Prepare input
mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(mel_spec_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_emotion = torch.argmax(probabilities, dim=1)

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
print(f"Predicted emotion: {emotions[predicted_emotion.item()]}")
print(f"Confidence: {probabilities[0].max().item():.2%}")
```

---

## Phase 1: Model Enhancement

### Step 1: Feature Extraction Pipeline

```python
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
        self.scaler = StandardScaler()
    
    def extract_mfcc(self, y, n_mfcc=13):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=n_mfcc)
        return mfcc.T  # (time_steps, 13)
    
    def extract_mel_spectrogram(self, y, n_mels=64):
        """Extract Mel-Spectrogram"""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T  # (time_steps, 64)
    
    def extract_zcr(self, y, frame_length=2048, hop_length=512):
        """Extract Zero Crossing Rate"""
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, 
                                                  hop_length=hop_length)
        return zcr.T  # (time_steps, 1)
    
    def extract_rmse(self, y, frame_length=2048, hop_length=512):
        """Extract RMSE Energy"""
        rmse = librosa.feature.rms(y=y, frame_length=frame_length, 
                                   hop_length=hop_length)
        return rmse.T  # (time_steps, 1)
    
    def extract_chroma_stft(self, y, n_chroma=12):
        """Extract Chroma STFT"""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_chroma=n_chroma)
        return chroma.T  # (time_steps, 12)
    
    def extract_all_features(self, y):
        """Extract all features and combine"""
        mfcc = self.extract_mfcc(y)
        mel_spec = self.extract_mel_spectrogram(y)
        zcr = self.extract_zcr(y)
        rmse = self.extract_rmse(y)
        chroma = self.extract_chroma_stft(y)
        
        # Combine all features
        features = np.concatenate([mfcc, mel_spec, zcr, rmse, chroma], axis=1)
        return features  # (time_steps, 13+64+1+1+12=91)
    
    def normalize_features(self, features):
        """Normalize features to mean=0, std=1"""
        return self.scaler.fit_transform(features)

# Usage
extractor = FeatureExtractor()
y, sr = librosa.load('audio.wav', sr=22050)
features = extractor.extract_all_features(y)
normalized_features = extractor.normalize_features(features)
print(f"Feature shape: {normalized_features.shape}")
```

### Step 2: Bi-LSTM Branch Implementation

```python
import torch
import torch.nn as nn

class BiLSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.5):
        super(BiLSTMBranch, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, 2*hidden_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state from both directions
        # h_n shape: (num_layers*2, batch_size, hidden_size)
        # Concatenate forward and backward last states
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 2*hidden_size)
        
        output = self.dropout(last_hidden)
        return output

# Usage
input_size = 91  # Total features
hidden_size = 128
lstm_branch = BiLSTMBranch(input_size, hidden_size)

# Test with sample input
batch_size = 32
seq_len = 100
x = torch.randn(batch_size, seq_len, input_size)
output = lstm_branch(x)
print(f"LSTM output shape: {output.shape}")  # (32, 256)
```

### Step 3: CNN Branch (Depthwise Separable)

```python
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNBranch(nn.Module):
    def __init__(self, input_channels=3, num_classes=8):
        super(CNNBranch, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Depthwise Separable Blocks
        self.block1 = DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1)
        self.block2 = DepthwiseSeparableConv(128, 256, kernel_size=3, padding=1)
        self.block3 = DepthwiseSeparableConv(256, 512, kernel_size=3, padding=1)
        
        # Global Average Pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 64, time_steps)
        Returns:
            output: (batch_size, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.block1(x)
        x = self.maxpool(x)
        
        x = self.block2(x)
        x = self.maxpool(x)
        
        x = self.block3(x)
        x = self.maxpool(x)
        
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        
        return x

# Usage
cnn_branch = CNNBranch(input_channels=3)
x = torch.randn(32, 3, 64, 100)  # (batch, channels, mel_bands, time)
output = cnn_branch(x)
print(f"CNN output shape: {output.shape}")  # (32, 512)
```

### Step 4: Ensemble Model

```python
class EnsembleClassifier(nn.Module):
    def __init__(self, cnn_feat_dim=512, lstm_feat_dim=256, num_classes=8):
        super(EnsembleClassifier, self).__init__()
        
        self.cnn_branch = CNNBranch(input_channels=3)
        self.lstm_branch = BiLSTMBranch(input_size=91, hidden_size=128)
        
        # Fusion layer
        fusion_input_dim = cnn_feat_dim + lstm_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, mel_spec, hand_crafted_features):
        """
        Args:
            mel_spec: (batch_size, 3, 64, time_steps)
            hand_crafted_features: (batch_size, seq_len, 91)
        Returns:
            logits: (batch_size, 8)
        """
        # CNN branch
        cnn_features = self.cnn_branch(mel_spec)
        
        # LSTM branch
        lstm_features = self.lstm_branch(hand_crafted_features)
        
        # Fusion
        fused = torch.cat([cnn_features, lstm_features], dim=1)
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits

# Usage
model = EnsembleClassifier()
mel_spec = torch.randn(32, 3, 64, 100)
hand_crafted = torch.randn(32, 100, 91)
logits = model(mel_spec, hand_crafted)
probabilities = torch.softmax(logits, dim=1)
print(f"Output shape: {logits.shape}")  # (32, 8)
print(f"Probabilities sum: {probabilities.sum(dim=1)}")  # Should be ~1.0
```

### Step 5: Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_ensemble(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the ensemble model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for mel_spec, hand_crafted, labels in train_loader:
            mel_spec = mel_spec.to(device)
            hand_crafted = hand_crafted.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(mel_spec, hand_crafted)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for mel_spec, hand_crafted, labels in val_loader:
                mel_spec = mel_spec.to(device)
                hand_crafted = hand_crafted.to(device)
                labels = labels.to(device)
                
                logits = model(mel_spec, hand_crafted)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ensemble_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping triggered")
            break
    
    return model

# Usage
# train_ensemble(model, train_loader, val_loader)
```

### Step 6: Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for mel_spec, hand_crafted, labels in test_loader:
            mel_spec = mel_spec.to(device)
            hand_crafted = hand_crafted.to(device)
            
            logits = model(mel_spec, hand_crafted)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
```

---

## Phase 2: Frontend Development

### Step 1: React Component for Microphone Recording

```javascript
import React, { useState, useRef } from 'react';

const AudioRecorder = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const [audioLevel, setAudioLevel] = useState(0);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Setup audio context for visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      // Setup media recorder
      mediaRecorderRef.current = new MediaRecorder(stream);
      const chunks = [];
      
      mediaRecorderRef.current.ondataavailable = (e) => {
        chunks.push(e.data);
      };
      
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioBlob(blob);
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      // Visualize audio level
      visualizeAudioLevel();
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const visualizeAudioLevel = () => {
    if (!analyserRef.current) return;
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
    setAudioLevel(average);
    
    if (isRecording) {
      requestAnimationFrame(visualizeAudioLevel);
    }
  };

  return (
    <div className="audio-recorder">
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </button>
      
      <div className="audio-level">
        <div className="level-bar" style={{ width: `${audioLevel}%` }}></div>
      </div>
      
      {audioBlob && (
        <div>
          <audio controls src={URL.createObjectURL(audioBlob)} />
          <button onClick={() => sendAudioToBackend(audioBlob)}>
            Analyze Emotion
          </button>
        </div>
      )}
    </div>
  );
};

export default AudioRecorder;
```

### Step 2: Emotion Display Component

```javascript
import React from 'react';

const EmotionDisplay = ({ prediction }) => {
  const emotions = [
    { name: 'Neutral', color: '#808080', icon: '😐' },
    { name: 'Calm', color: '#87CEEB', icon: '😌' },
    { name: 'Happy', color: '#FFD700', icon: '😊' },
    { name: 'Sad', color: '#4169E1', icon: '😢' },
    { name: 'Angry', color: '#FF4500', icon: '😠' },
    { name: 'Fearful', color: '#9932CC', icon: '😨' },
    { name: 'Disgust', color: '#228B22', icon: '🤢' },
    { name: 'Surprised', color: '#FF69B4', icon: '😲' }
  ];

  if (!prediction) return <div>No prediction yet</div>;

  const predictedEmotion = emotions[prediction.emotion_index];
  const confidence = (prediction.confidence * 100).toFixed(1);

  return (
    <div className="emotion-display">
      <div className="predicted-emotion" style={{ backgroundColor: predictedEmotion.color }}>
        <div className="emotion-icon">{predictedEmotion.icon}</div>
        <div className="emotion-name">{predictedEmotion.name}</div>
        <div className="confidence">Confidence: {confidence}%</div>
      </div>

      <div className="emotion-probabilities">
        {emotions.map((emotion, idx) => (
          <div key={idx} className="emotion-bar">
            <span className="emotion-label">{emotion.name}</span>
            <div className="bar-container">
              <div 
                className="bar-fill" 
                style={{
                  width: `${prediction.probabilities[idx] * 100}%`,
                  backgroundColor: emotion.color
                }}
              />
            </div>
            <span className="probability">
              {(prediction.probabilities[idx] * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EmotionDisplay;
```

---

## Phase 3: API Integration

### Step 1: Flask Backend API

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnsembleClassifier().to(device)
model.load_state_dict(torch.load('best_ensemble_model.pth'))
model.eval()

# Feature extractor
extractor = FeatureExtractor()

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_data = io.BytesIO(audio_file.read())
        
        # Load audio
        y, sr = librosa.load(audio_data, sr=22050)
        
        # Extract features
        mel_spec = extractor.extract_mel_spectrogram(y)
        hand_crafted = extractor.extract_all_features(y)
        
        # Normalize
        hand_crafted = extractor.normalize_features(hand_crafted)
        
        # Prepare tensors
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
        hand_crafted_tensor = torch.FloatTensor(hand_crafted).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(mel_spec_tensor, hand_crafted_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
        
        return jsonify({
            'emotion': emotions[predicted_idx],
            'emotion_index': predicted_idx,
            'confidence': float(confidence),
            'probabilities': probabilities[0].cpu().numpy().tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## Troubleshooting Guide

### Issue: Model accuracy is low

**Solution**:
1. Check feature extraction is correct
2. Verify data preprocessing (normalization, resampling)
3. Increase training epochs
4. Adjust learning rate
5. Check for class imbalance in training data

### Issue: Out of memory during training

**Solution**:
1. Reduce batch size
2. Use gradient accumulation
3. Use mixed precision training
4. Reduce model size

### Issue: Audio recording not working

**Solution**:
1. Check microphone permissions
2. Verify browser compatibility
3. Check audio context initialization
4. Test with different audio formats

---

**Document Version**: 1.0  
**Created**: 2024-01-15  
**Status**: Complete
