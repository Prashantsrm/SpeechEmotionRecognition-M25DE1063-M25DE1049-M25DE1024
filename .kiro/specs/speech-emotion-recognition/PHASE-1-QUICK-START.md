# Phase 1 Quick Start: Model Enhancement
## Week 1-3 Implementation Guide

---

## 🎯 Phase 1 Goal

**Improve model accuracy from 94% to >95%** by implementing an ensemble architecture combining CNN and Bi-LSTM with hand-crafted audio features.

---

## 📋 Phase 1 Tasks Overview

| Task | Duration | Status |
|------|----------|--------|
| 1.1 Implement Bi-LSTM Branch | 2 days | ⏳ To Start |
| 1.2 Implement Feature Extraction | 3 days | ⏳ To Start |
| 1.3 Implement Ensemble Model | 2 days | ⏳ To Start |
| 1.4 Integrate with Existing CNN | 2 days | ⏳ To Start |
| 1.5 Implement Training Pipeline | 2 days | ⏳ To Start |
| 1.6 Train Ensemble Model | 3 days | ⏳ To Start |
| 1.7 Evaluate on RAVDESS Test Set | 2 days | ⏳ To Start |
| 1.8 Cross-Dataset Validation | 2 days | ⏳ To Start |

**Total**: 18 days (3 weeks)

---

## 🚀 Day 1-2: Setup & Bi-LSTM Implementation

### Step 1: Setup Development Environment

```bash
# Create project directory
mkdir speech-emotion-recognition
cd speech-emotion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchaudio librosa numpy scikit-learn
pip install transformers datasets
pip install matplotlib seaborn jupyter
pip install flask flask-cors
```

### Step 2: Create Project Structure

```bash
mkdir -p src/{models,features,training,evaluation}
mkdir -p notebooks
mkdir -p data
mkdir -p models
mkdir -p results
```

### Step 3: Implement Bi-LSTM Branch

**File**: `src/models/lstm_branch.py`

```python
import torch
import torch.nn as nn

class BiLSTMBranch(nn.Module):
    """Bi-directional LSTM for temporal feature processing"""
    
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate forward and backward last hidden states
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        output = self.dropout(last_hidden)
        
        return output

# Test the module
if __name__ == "__main__":
    lstm = BiLSTMBranch(input_size=91, hidden_size=128)
    x = torch.randn(32, 100, 91)  # (batch, seq_len, features)
    output = lstm(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 256)
```

### Step 4: Test Bi-LSTM

```bash
python src/models/lstm_branch.py
# Expected output: Output shape: torch.Size([32, 256])
```

---

## 🎵 Day 3-5: Feature Extraction Implementation

### Step 1: Implement Feature Extractor

**File**: `src/features/extractor.py`

```python
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """Extract hand-crafted audio features"""
    
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
        return features  # (time_steps, 91)
    
    def normalize_features(self, features):
        """Normalize features to mean=0, std=1"""
        return self.scaler.fit_transform(features)

# Test the extractor
if __name__ == "__main__":
    import librosa
    
    extractor = FeatureExtractor()
    
    # Load sample audio
    y, sr = librosa.load('sample_audio.wav', sr=22050)
    
    # Extract features
    features = extractor.extract_all_features(y)
    print(f"Feature shape: {features.shape}")  # Should be (time_steps, 91)
    
    # Normalize
    normalized = extractor.normalize_features(features)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
```

### Step 2: Test Feature Extraction

```bash
# Download sample audio if needed
# Then run:
python src/features/extractor.py
```

---

## 🧠 Day 6-7: Ensemble Model Implementation

### Step 1: Implement CNN Branch

**File**: `src/models/cnn_branch.py`

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
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
    """CNN branch for spatial feature extraction"""
    
    def __init__(self, input_channels=3, num_classes=8):
        super(CNNBranch, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1)
        self.block2 = DepthwiseSeparableConv(128, 256, kernel_size=3, padding=1)
        self.block3 = DepthwiseSeparableConv(256, 512, kernel_size=3, padding=1)
        
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
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
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        return x
```

### Step 2: Implement Ensemble Model

**File**: `src/models/ensemble.py`

```python
import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .lstm_branch import BiLSTMBranch

class EnsembleClassifier(nn.Module):
    """Ensemble model combining CNN and Bi-LSTM"""
    
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
        cnn_features = self.cnn_branch(mel_spec)
        lstm_features = self.lstm_branch(hand_crafted_features)
        
        fused = torch.cat([cnn_features, lstm_features], dim=1)
        fused = self.fusion(fused)
        
        logits = self.classifier(fused)
        return logits

# Test the ensemble
if __name__ == "__main__":
    model = EnsembleClassifier()
    mel_spec = torch.randn(32, 3, 64, 100)
    hand_crafted = torch.randn(32, 100, 91)
    logits = model(mel_spec, hand_crafted)
    print(f"Output shape: {logits.shape}")  # Should be (32, 8)
```

---

## 📊 Day 8-10: Training Implementation

### Step 1: Implement Training Pipeline

**File**: `src/training/trainer.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_ensemble(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the ensemble model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    
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
            
            logits = model(mel_spec, hand_crafted)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ensemble_model.pth')
    
    return model
```

---

## ✅ Day 11-14: Evaluation

### Step 1: Implement Evaluation Metrics

**File**: `src/evaluation/metrics.py`

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for mel_spec, hand_crafted, labels in test_loader:
            mel_spec = mel_spec.to(device)
            hand_crafted = hand_crafted.to(device)
            
            logits = model(mel_spec, hand_crafted)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
```

---

## 📅 Phase 1 Timeline

| Week | Days | Tasks | Deliverables |
|------|------|-------|--------------|
| Week 1 | 1-2 | Setup, Bi-LSTM | BiLSTMBranch module |
| Week 1 | 3-5 | Feature Extraction | FeatureExtractor module |
| Week 2 | 6-7 | Ensemble Model | EnsembleClassifier module |
| Week 2 | 8-10 | Training Pipeline | Training scripts |
| Week 3 | 11-14 | Evaluation | Evaluation metrics, results |

---

## 🎯 Success Criteria for Phase 1

- [ ] Bi-LSTM branch implemented and tested
- [ ] Feature extraction pipeline working
- [ ] Ensemble model architecture complete
- [ ] Training pipeline functional
- [ ] Model accuracy ≥95% on RAVDESS test set
- [ ] Cross-dataset validation completed
- [ ] All code documented
- [ ] Results saved and analyzed

---

## 📝 Deliverables for Phase 1

1. **Code Files**:
   - `src/models/lstm_branch.py`
   - `src/models/cnn_branch.py`
   - `src/models/ensemble.py`
   - `src/features/extractor.py`
   - `src/training/trainer.py`
   - `src/evaluation/metrics.py`

2. **Models**:
   - `models/best_ensemble_model.pth`
   - `models/ensemble_model_final.pth`

3. **Results**:
   - `results/evaluation_report.txt`
   - `results/confusion_matrix.png`
   - `results/training_curves.png`

4. **Documentation**:
   - Phase 1 completion report
   - Model architecture documentation
   - Training results summary

---

## 🚀 Next Steps After Phase 1

Once Phase 1 is complete:
1. Review evaluation results
2. Compare with baseline CNN model
3. Analyze cross-dataset performance
4. Begin Phase 2 (Frontend Development)

---

**Good luck with Phase 1! 🎉**

For detailed information, refer to:
- IMPLEMENTATION-GUIDE.md - Code examples
- MATHEMATICAL-FORMULAS.md - Formulas and explanations
- design.md - Architecture details
- tasks.md - Complete task list

