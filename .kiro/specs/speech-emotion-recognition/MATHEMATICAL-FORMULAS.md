# Mathematical Formulas & Step-by-Step Explanations
## Enhanced Speech Emotion Recognition System

---

## Table of Contents
1. [Audio Preprocessing](#audio-preprocessing)
2. [Feature Extraction](#feature-extraction)
3. [Model Architecture](#model-architecture)
4. [Training & Optimization](#training--optimization)
5. [Evaluation Metrics](#evaluation-metrics)

---

## Audio Preprocessing

### 1. Audio Normalization

**What it does**: Scales audio to a standard range so all audio files are treated equally.

**Formula**:
```
x_normalized = x / max(|x|)
```

**Explanation**:
- `x` = original audio signal (array of numbers)
- `max(|x|)` = the largest absolute value in the audio
- Result: Audio values between -1 and +1

**Example**:
```
Original audio: [0.5, 1.2, -0.8, 0.3]
max(|x|) = 1.2
Normalized: [0.417, 1.0, -0.667, 0.25]
```

### 2. Resampling

**What it does**: Converts audio to a standard sample rate (22050 Hz).

**Formula**:
```
new_sample_rate = 22050 Hz
resampling_factor = new_sample_rate / original_sample_rate
```

**Explanation**:
- If original audio is 44100 Hz, we take every 2nd sample
- If original audio is 16000 Hz, we interpolate between samples
- Result: All audio has same sample rate for consistent processing

---

## Feature Extraction

### 1. MFCC (Mel-Frequency Cepstral Coefficients)

**What it does**: Extracts 13 numbers that represent the "sound signature" of speech, mimicking how human ears perceive sound.

**Step-by-Step Process**:

#### Step 1: Divide audio into frames
```
frame_length = 2048 samples
hop_length = 512 samples
n_frames = (len(audio) - frame_length) / hop_length
```

**Example**: 1-second audio at 22050 Hz = 22050 samples
- Number of frames = (22050 - 2048) / 512 ≈ 39 frames

#### Step 2: Apply Hamming Window
```
window(n) = 0.54 - 0.46 * cos(2π * n / (N-1))
where n = 0, 1, ..., N-1
```

**What it does**: Smooths the edges of each frame to reduce artifacts.

#### Step 3: Compute Power Spectrogram
```
X(k) = |FFT(frame * window)|²
```

**What it does**: Converts time-domain audio to frequency-domain (shows which frequencies are present).

#### Step 4: Apply Mel-Scale Filter Banks
```
mel_scale(f) = 2595 * log10(1 + f/700)
```

**What it does**: Converts frequency to mel-scale (how humans perceive pitch).

**Example**:
- 100 Hz → 150 mels
- 1000 Hz → 1000 mels
- 10000 Hz → 2840 mels

#### Step 5: Compute Log Power
```
log_power = log(mel_spectrogram + ε)
where ε = 1e-10 (small value to avoid log(0))
```

#### Step 6: Apply Discrete Cosine Transform (DCT)
```
MFCC(i) = Σ(j=0 to n_mels-1) log_power(j) * cos(π * i * (j + 0.5) / n_mels)
```

**Result**: 13 MFCC coefficients per frame

**Visual Representation**:
```
Audio Signal → Frames → Power Spectrum → Mel-Scale → Log → DCT → MFCC (13 values)
```

### 2. Mel-Spectrogram

**What it does**: Creates a 2D image of sound showing frequency content over time.

**Formula**:
```
mel_spectrogram(t, f) = log(|STFT(t, f)|² + ε)
where:
- t = time frame
- f = frequency bin (converted to mel-scale)
- STFT = Short-Time Fourier Transform
```

**Dimensions**: (64 mel-bands, time_steps)

**Example**:
- 3-second audio at 22050 Hz
- Produces: (64, 130) matrix
- Each row = one frequency band
- Each column = one time frame

### 3. Zero Crossing Rate (ZCR)

**What it does**: Counts how many times the audio signal crosses zero (changes from positive to negative or vice versa). High ZCR = noisy/unvoiced, Low ZCR = voiced/clean.

**Formula**:
```
ZCR(t) = (1/2) * Σ(n=0 to N-1) |sign(x(n)) - sign(x(n+1))|
where:
- sign(x) = +1 if x > 0, -1 if x < 0
- N = frame length
```

**Example**:
```
Audio frame: [0.1, -0.2, 0.3, -0.1, 0.2]
Sign sequence: [+, -, +, -, +]
Sign changes: 4 (between each pair)
ZCR = 4 / (2 * 5) = 0.4
```

**Interpretation**:
- ZCR ≈ 0.1 → Voiced sound (vowels)
- ZCR ≈ 0.5 → Unvoiced sound (consonants)
- ZCR ≈ 0.8 → Noise

### 4. RMSE (Root Mean Square Energy)

**What it does**: Measures the loudness/energy of the audio signal.

**Formula**:
```
RMSE(t) = √(1/N * Σ(n=0 to N-1) x(n)²)
where:
- x(n) = audio sample
- N = frame length
```

**Example**:
```
Audio frame: [0.1, 0.2, -0.15, 0.3]
Squares: [0.01, 0.04, 0.0225, 0.09]
Mean: (0.01 + 0.04 + 0.0225 + 0.09) / 4 = 0.0406
RMSE = √0.0406 ≈ 0.201
```

**Interpretation**:
- RMSE ≈ 0.01 → Quiet/silence
- RMSE ≈ 0.1 → Normal speech
- RMSE ≈ 0.3 → Loud speech

### 5. Chroma STFT

**What it does**: Extracts 12 pitch-related features (one for each musical note: C, C#, D, D#, E, F, F#, G, G#, A, A#, B).

**Formula**:
```
Chroma(c) = Σ(f: f ≡ c mod 12) |STFT(f)|
where:
- c = chroma bin (0-11)
- f = frequency bin
```

**Result**: 12 values per frame representing pitch content

---

## Feature Normalization

**What it does**: Scales all features to have mean=0 and standard deviation=1, so no single feature dominates.

**Formula**:
```
x_normalized = (x - mean(x)) / std(x)
where:
- mean(x) = average of all values
- std(x) = standard deviation
```

**Example**:
```
Original MFCC: [10, 20, 15, 25, 30]
mean = 20
std = 7.07
Normalized: [-1.41, 0, -0.71, 0.71, 1.41]
```

**Property**: Normalized values are typically between -3 and +3

---

## Model Architecture

### 1. CNN Branch - Depthwise Separable Convolution

**What it does**: Efficiently extracts spatial features from mel-spectrograms.

#### Standard Convolution (for comparison):
```
output(i,j) = Σ(m,n) kernel(m,n) * input(i+m, j+n) + bias
```

**Parameters**: (kernel_height × kernel_width × input_channels × output_channels)

#### Depthwise Separable Convolution:

**Step 1: Depthwise Convolution**
```
output_depthwise(i,j,c) = Σ(m,n) kernel_depthwise(m,n,c) * input(i+m, j+n, c) + bias
```

**Parameters**: (kernel_height × kernel_width × input_channels)

**Step 2: Pointwise Convolution (1×1)**
```
output_pointwise(i,j,c') = Σ(c) kernel_pointwise(1,1,c,c') * output_depthwise(i,j,c) + bias
```

**Parameters**: (input_channels × output_channels)

**Advantage**: 
- Standard: 7×7×3×64 = 9,408 parameters
- Depthwise Separable: (7×7×3) + (3×64) = 339 parameters
- **Reduction: 97%!**

### 2. CNN Architecture Layers

```
Input: (3, 64, T) - 3 channels, 64 mel-bands, T time steps

Layer 1: Conv 7×7, 64 filters, stride=2
Output: (64, 31, T/2)

Layer 2-4: Depthwise Separable Blocks
  Block 1: 64 → 128 channels
  Block 2: 128 → 256 channels
  Block 3: 256 → 512 channels
  Each with: BatchNorm, ReLU, MaxPool

Global Average Pooling:
Output: (512,) - 512-dimensional feature vector
```

### 3. Bi-LSTM Branch

**What it does**: Processes sequences of features in both forward and backward directions to capture temporal context.

#### LSTM Cell Equations:

```
Input gate:     i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
Forget gate:    f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
Cell gate:      g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
Output gate:    o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)

Cell state:     c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden state:   h_t = o_t ⊙ tanh(c_t)

where:
- σ = sigmoid function (0 to 1)
- tanh = hyperbolic tangent (-1 to 1)
- ⊙ = element-wise multiplication
- W = weight matrices
- b = bias vectors
```

**Bi-LSTM**:
```
Forward pass:  h_t^→ = LSTM(x_t, h_{t-1}^→)
Backward pass: h_t^← = LSTM(x_t, h_{t+1}^←)
Concatenate:   h_t = [h_t^→; h_t^←]
```

**Result**: 256-dimensional feature vector (128 forward + 128 backward)

### 4. Feature Fusion

**What it does**: Combines CNN and LSTM features.

```
fused = concatenate([cnn_features, lstm_features])
fused = Dense(256)(fused)  # 768 → 256
fused = ReLU(fused)
fused = Dropout(0.5)(fused)
```

### 5. Classification Head

```
logits = Dense(128)(fused)  # 256 → 128
logits = ReLU(logits)
logits = Dropout(0.5)(logits)
logits = Dense(8)(logits)   # 128 → 8 (emotions)

probabilities = Softmax(logits)
```

### 6. Softmax Function

**What it does**: Converts raw scores to probabilities that sum to 1.

**Formula**:
```
P(emotion_i) = exp(logit_i) / Σ(j=0 to 7) exp(logit_j)
```

**Example**:
```
Raw logits: [2.0, 1.0, 0.5, -1.0, 0.2, -0.5, 1.5, 0.8]

exp(logits): [7.39, 2.72, 1.65, 0.37, 1.22, 0.61, 4.48, 2.23]
Sum: 20.67

Probabilities: [0.357, 0.132, 0.080, 0.018, 0.059, 0.030, 0.217, 0.108]
Sum: 1.0 ✓

Predicted emotion: Index 0 (highest probability)
Confidence: 0.357 (35.7%)
```

---

## Training & Optimization

### 1. Cross-Entropy Loss

**What it does**: Measures how wrong the model's predictions are.

**Formula**:
```
Loss = -Σ(i=0 to 7) y_i * log(p_i)
where:
- y_i = 1 if true emotion is i, 0 otherwise (one-hot encoding)
- p_i = predicted probability for emotion i
```

**Example**:
```
True emotion: Angry (index 1)
True labels: [0, 1, 0, 0, 0, 0, 0, 0]

Predicted probabilities: [0.1, 0.7, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]

Loss = -(0*log(0.1) + 1*log(0.7) + 0*log(0.05) + ... + 0*log(0.01))
     = -log(0.7)
     = 0.357

Lower loss = better predictions
```

### 2. Adam Optimizer

**What it does**: Updates model weights to minimize loss.

**Formulas**:
```
m_t = β1 * m_{t-1} + (1 - β1) * g_t          (momentum)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²        (velocity)

m_hat_t = m_t / (1 - β1^t)                   (bias correction)
v_hat_t = v_t / (1 - β2^t)                   (bias correction)

θ_t = θ_{t-1} - α * m_hat_t / (√v_hat_t + ε)

where:
- g_t = gradient
- β1 = 0.9 (momentum coefficient)
- β2 = 0.999 (velocity coefficient)
- α = learning rate (0.001)
- ε = 1e-8 (small value)
```

**Intuition**: Like a ball rolling downhill with momentum, but also considering the terrain (velocity).

### 3. Learning Rate Scheduling

**ReduceLROnPlateau**:
```
if validation_loss doesn't improve for 5 epochs:
    learning_rate = learning_rate * 0.5
    
Example:
Epoch 1-5: lr = 0.001
Epoch 6-10: lr = 0.0005 (no improvement)
Epoch 11-15: lr = 0.00025 (still no improvement)
```

### 4. Dropout Regularization

**What it does**: Randomly disables 50% of neurons during training to prevent overfitting.

```
During training:
output = input * random_mask (50% zeros)

During inference:
output = input * 0.5 (scale by keep probability)
```

**Effect**: Model learns more robust features that don't rely on specific neurons.

---

## Evaluation Metrics

### 1. Accuracy

**Formula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
where:
- TP = True Positives (correctly predicted emotion)
- TN = True Negatives (correctly rejected emotion)
- FP = False Positives (incorrectly predicted emotion)
- FN = False Negatives (missed emotion)
```

**Example**:
```
Test set: 100 audio samples
Correct predictions: 95
Incorrect predictions: 5

Accuracy = 95 / 100 = 0.95 (95%)
```

### 2. Precision & Recall (per emotion)

**Precision** (How many predicted emotions are correct?):
```
Precision = TP / (TP + FP)
```

**Recall** (How many actual emotions did we find?):
```
Recall = TP / (TP + FN)
```

**Example** (for "Happy" emotion):
```
Predicted Happy: 50 samples
Actually Happy: 45 samples
Missed Happy: 5 samples

Precision = 45 / 50 = 0.90 (90% of predicted Happy are correct)
Recall = 45 / (45 + 5) = 0.90 (we found 90% of actual Happy)
```

### 3. F1-Score

**Formula**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interpretation**: Harmonic mean of precision and recall (0 to 1, higher is better)

**Example**:
```
Precision = 0.90
Recall = 0.85

F1 = 2 * (0.90 * 0.85) / (0.90 + 0.85)
   = 2 * 0.765 / 1.75
   = 0.874
```

### 4. Confusion Matrix

**What it shows**: For each emotion, how many were correctly classified vs misclassified.

**Example** (8×8 matrix):
```
                Predicted
              N  C  H  S  A  F  D  Su
Actual    N  [45  2  1  0  1  0  1  0]
          C  [ 1 48  0  1  0  0  0  0]
          H  [ 0  0 50  0  0  0  0  0]
          S  [ 1  0  0 47  1  0  1  0]
          A  [ 0  0  0  1 49  0  0  0]
          F  [ 0  0  0  0  0 48  1  1]
          D  [ 0  0  0  1  0  1 47  1]
          Su [ 0  0  0  0  0  0  1 49]

Diagonal = correct predictions
Off-diagonal = misclassifications
```

### 5. AUC-ROC (Area Under Receiver Operating Characteristic Curve)

**What it measures**: How well the model separates one emotion from all others.

**Formula**:
```
AUC-ROC = Area under the curve of (True Positive Rate vs False Positive Rate)

TPR = TP / (TP + FN)  (sensitivity)
FPR = FP / (FP + TN)  (1 - specificity)
```

**Interpretation**:
- AUC = 1.0 → Perfect classifier
- AUC = 0.5 → Random classifier
- AUC = 0.0 → Worst classifier

---

## Summary Table

| Component | Formula | Output | Purpose |
|-----------|---------|--------|---------|
| MFCC | DCT(log(mel_spectrogram)) | 13 values | Spectral features |
| Mel-Spec | log(\|STFT\|²) | (64, T) matrix | Time-frequency representation |
| ZCR | sign changes / frame length | 1 value | Voicing indicator |
| RMSE | √(mean(x²)) | 1 value | Energy indicator |
| Chroma | Pitch aggregation | 12 values | Pitch content |
| CNN | Depthwise separable conv | 512 values | Spatial features |
| LSTM | Bidirectional processing | 256 values | Temporal features |
| Softmax | exp(x) / Σexp(x) | 8 probabilities | Emotion probabilities |
| Loss | -Σ(y * log(p)) | 1 value | Training error |
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | 0-1 | Classification accuracy |

---

## Visual Learning Guide

### How MFCC Works (Step-by-Step)

```
1. Audio Signal
   ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁

2. Divide into Frames
   [Frame 1] [Frame 2] [Frame 3] ...

3. Power Spectrum (Frequency Content)
   Frequency
   ▲
   │     ▁▂▃▂▁
   │   ▁▃▅▇█▇▅▃▁
   │ ▁▃▅▇█████▇▅▃▁
   └─────────────────► Frequency Bins

4. Mel-Scale (Human Perception)
   Mel
   ▲
   │     ▁▂▃▂▁
   │   ▁▃▅▇█▇▅▃▁
   │ ▁▃▅▇█████▇▅▃▁
   └─────────────────► Mel Bins

5. Log Power (Compress Range)
   Log Power
   ▲
   │     ▁▂▃▂▁
   │   ▁▃▅▇█▇▅▃▁
   │ ▁▃▅▇█████▇▅▃▁
   └─────────────────► Mel Bins

6. DCT (Extract Features)
   MFCC Coefficients: [c1, c2, c3, ..., c13]
```

### How Softmax Works

```
Raw Scores:        [2.0, 1.0, 0.5, -1.0, 0.2, -0.5, 1.5, 0.8]
                    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
Exponential:       [7.4, 2.7, 1.7, 0.4, 1.2, 0.6, 4.5, 2.2]
                    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
Normalize:         [0.36, 0.13, 0.08, 0.02, 0.06, 0.03, 0.22, 0.11]
                    ↓
Predicted Emotion: Index 0 (Neutral) with 36% confidence
```

---

## Key Takeaways

1. **Features capture different aspects of emotion**:
   - MFCC: What frequencies are present
   - ZCR: Whether sound is voiced or unvoiced
   - RMSE: How loud the sound is
   - Chroma: What pitch content is present

2. **Ensemble combines two approaches**:
   - CNN: Learns spatial patterns in spectrograms
   - LSTM: Learns temporal patterns in sequences

3. **Training optimizes model to minimize loss**:
   - Adam optimizer adjusts weights
   - Learning rate scheduler prevents overfitting
   - Dropout adds regularization

4. **Evaluation metrics measure different aspects**:
   - Accuracy: Overall correctness
   - Precision/Recall: Per-emotion performance
   - F1-Score: Balanced metric
   - AUC-ROC: Discrimination ability

---

**Document Version**: 1.0  
**Created**: 2024-01-15  
**Status**: Complete
