# Hugging Face Integration Guide
## Using Pre-Trained Models for Speech Emotion Recognition

---

## Overview

Hugging Face provides pre-trained audio models that can significantly improve your emotion recognition system. This guide shows how to integrate them.

---

## Installation

```bash
pip install transformers datasets librosa torch torchaudio
```

---

## Option 1: Using Wav2Vec2 for Feature Extraction

### What is Wav2Vec2?

Wav2Vec2 is a self-supervised model trained on 53,000 hours of unlabeled speech. It learns powerful audio representations without labeled data.

### Step 1: Load Pre-Trained Model

```python
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
```

### Step 2: Extract Features

```python
def extract_wav2vec2_features(audio_path):
    """Extract Wav2Vec2 features from audio"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)  # Wav2Vec2 expects 16kHz
    
    # Process audio
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        last_hidden_state = outputs.last_hidden_state  # (1, time_steps, 768)
    
    return last_hidden_state

# Usage
features = extract_wav2vec2_features('audio.wav')
print(f"Feature shape: {features.shape}")  # (1, time_steps, 768)
```

### Step 3: Use with Ensemble Model

```python
class EnsembleWithWav2Vec2(nn.Module):
    def __init__(self, num_classes=8):
        super(EnsembleWithWav2Vec2, self).__init__()
        
        # Wav2Vec2 feature extractor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, audio_waveform):
        """
        Args:
            audio_waveform: (batch_size, audio_length)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Extract Wav2Vec2 features
        inputs = self.processor(audio_waveform, sampling_rate=16000, 
                               return_tensors="pt", padding=True)
        with torch.no_grad():
            wav2vec2_features = self.wav2vec2(**inputs).last_hidden_state
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(wav2vec2_features)
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Classification
        logits = self.classifier(last_hidden)
        return logits

# Usage
model = EnsembleWithWav2Vec2()
audio = torch.randn(32, 16000)  # 1 second at 16kHz
logits = model(audio)
print(f"Output shape: {logits.shape}")  # (32, 8)
```

---

## Option 2: Using HuBERT for Feature Extraction

### What is HuBERT?

HuBERT (Hidden Unit BERT) is another self-supervised model trained on 960 hours of labeled speech data. It's often better for downstream tasks.

### Step 1: Load HuBERT

```python
from transformers import HubertModel, Wav2Vec2Processor

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
```

### Step 2: Extract Features

```python
def extract_hubert_features(audio_path):
    """Extract HuBERT features from audio"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        last_hidden_state = outputs.last_hidden_state  # (1, time_steps, 768)
    
    return last_hidden_state

# Usage
features = extract_hubert_features('audio.wav')
print(f"Feature shape: {features.shape}")  # (1, time_steps, 768)
```

---

## Option 3: Using Hugging Face Datasets

### Load RAVDESS from Hugging Face

```python
from datasets import load_dataset

# Load RAVDESS dataset
dataset = load_dataset("ravdess", "speech")

print(f"Dataset structure: {dataset}")
print(f"Number of samples: {len(dataset['train'])}")

# Access a sample
sample = dataset['train'][0]
print(f"Sample keys: {sample.keys()}")
print(f"Audio shape: {sample['audio']['array'].shape}")
print(f"Emotion: {sample['emotion']}")
```

### Create DataLoader

```python
from torch.utils.data import DataLoader, Dataset

class RAVDESSDataset(Dataset):
    def __init__(self, dataset_split, processor, sr=16000):
        self.dataset = dataset_split
        self.processor = processor
        self.sr = sr
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get audio
        audio = sample['audio']['array']
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=self.sr, 
                               return_tensors="pt", padding=True)
        
        # Get emotion label
        emotion = sample['emotion']
        
        return {
            'input_values': inputs['input_values'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(emotion, dtype=torch.long)
        }

# Create dataset and dataloader
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
train_dataset = RAVDESSDataset(dataset['train'], processor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in train_loader:
    print(f"Input shape: {batch['input_values'].shape}")
    print(f"Label shape: {batch['label'].shape}")
    break
```

---

## Option 4: Fine-Tuning Pre-Trained Models

### Fine-Tune Wav2Vec2 for Emotion Recognition

```python
from transformers import Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer

# Load pre-trained model for classification
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=8,
    problem_type="single_label_classification"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-emotion-model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=5000,
    num_train_epochs=3,
    save_total_limit=2,
    push_to_hub=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model.save_pretrained("./wav2vec2-emotion-final")
processor.save_pretrained("./wav2vec2-emotion-final")
```

### Load Fine-Tuned Model

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Load fine-tuned model
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-emotion-final")
model = Wav2Vec2ForSequenceClassification.from_pretrained("./wav2vec2-emotion-final")

# Make prediction
def predict_emotion(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_id = torch.argmax(logits, dim=-1).item()
    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    return {
        'emotion': emotions[predicted_id],
        'confidence': float(probabilities[predicted_id]),
        'probabilities': probabilities.tolist()
    }

# Usage
result = predict_emotion('audio.wav')
print(result)
```

---

## Option 5: Deploy to Hugging Face Model Hub

### Step 1: Create Hugging Face Account

1. Go to https://huggingface.co
2. Create an account
3. Create a new model repository

### Step 2: Upload Model

```python
from huggingface_hub import HfApi, HfFolder

# Login
HfFolder.save_token("your_hugging_face_token")

# Upload model
api = HfApi()
api.upload_folder(
    folder_path="./wav2vec2-emotion-final",
    repo_id="your-username/wav2vec2-emotion-recognition",
    repo_type="model"
)
```

### Step 3: Load from Hub

```python
# Anyone can now load your model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "your-username/wav2vec2-emotion-recognition"
)
processor = Wav2Vec2Processor.from_pretrained(
    "your-username/wav2vec2-emotion-recognition"
)
```

---

## Comparison: Hand-Crafted vs Pre-Trained Features

| Aspect | Hand-Crafted | Wav2Vec2 | HuBERT |
|--------|--------------|----------|---------|
| Feature Dim | 91 | 768 | 768 |
| Training Data | None | 53k hours | 960 hours |
| Interpretability | High | Low | Low |
| Accuracy | ~94% | ~96% | ~97% |
| Inference Speed | Fast | Slower | Slower |
| Model Size | Small | Large | Large |
| Fine-Tuning | Not needed | Optional | Recommended |

---

## Recommended Approach

### For Best Accuracy:
```python
# Combine hand-crafted features with HuBERT
class HybridEnsemble(nn.Module):
    def __init__(self):
        super(HybridEnsemble, self).__init__()
        
        # Hand-crafted features branch
        self.hand_crafted_lstm = BiLSTMBranch(input_size=91, hidden_size=128)
        
        # HuBERT features branch
        self.hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_lstm = nn.LSTM(768, 256, 2, batch_first=True, bidirectional=True)
        
        # Fusion and classification
        self.fusion = nn.Linear(256 + 512, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 8)
        )
    
    def forward(self, hand_crafted_features, audio_waveform):
        # Hand-crafted branch
        hc_features = self.hand_crafted_lstm(hand_crafted_features)
        
        # HuBERT branch
        inputs = self.hubert_processor(audio_waveform, sampling_rate=16000, 
                                      return_tensors="pt", padding=True)
        hubert_out = self.hubert_model(**inputs).last_hidden_state
        lstm_out, (h_n, c_n) = self.hubert_lstm(hubert_out)
        hubert_features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Fusion
        fused = torch.cat([hc_features, hubert_features], dim=1)
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        return logits
```

---

## Troubleshooting

### Issue: Model too large for GPU

**Solution**:
```python
# Use quantization
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
model = model.half()  # Convert to FP16
```

### Issue: Slow inference

**Solution**:
```python
# Use ONNX for faster inference
from transformers.onnx import convert_pytorch_to_onnx

convert_pytorch_to_onnx(
    framework="pt",
    model_name_or_path="facebook/wav2vec2-base-960h",
    outputs_path="./onnx_model"
)
```

---

**Document Version**: 1.0  
**Created**: 2024-01-15  
**Status**: Complete
