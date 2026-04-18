# Enhanced Speech Emotion Recognition System - Design Summary

## Project Overview

This is a comprehensive technical design for enhancing an existing Speech Emotion Recognition (SER) system from IIT Jodhpur's Team X course. The project builds upon a CNN-based model achieving 94% accuracy on RAVDESS dataset and proposes an ensemble architecture combining CNN and Bi-LSTM with hand-crafted audio features, plus a real-time frontend interface.

**Team**: Asit Jain, Avinash Singh, Prashant Kumar Mishra  
**Institution**: IIT Jodhpur  
**Course**: Speech Understanding (Team X)

## Key Enhancement Goals

1. **Improve Model Accuracy**: Target >95% on RAVDESS, >90% on cross-dataset validation
2. **Optimize Efficiency**: Lightweight architecture suitable for edge deployment (<10 MB quantized)
3. **Real-Time Frontend**: Web-based interface for live voice input and emotion detection
4. **Research Integration**: Implement ensemble approach from "Speech emotion recognition with lightweight deep neural ensemble model using hand crafted features"

## System Architecture

### High-Level Components

```
Audio Input (Microphone/File)
    ↓
Audio Preprocessing (Noise removal, Normalization)
    ↓
Feature Extraction (MFCC, Mel-Spec, ZCR, RMSE, Chroma)
    ↓
Ensemble Model
    ├─ CNN Branch (Depthwise Separable Conv)
    └─ Bi-LSTM Branch (Bidirectional Processing)
    ↓
Feature Fusion & Classification
    ↓
Emotion Prediction (8 classes + confidence)
    ↓
Frontend Display & API Response
```

## Core Technical Components

### 1. Feature Extraction Pipeline
- **MFCC**: 13 coefficients capturing spectral characteristics
- **Mel-Spectrogram**: 64 mel-bands, dB scale representation
- **Zero Crossing Rate (ZCR)**: Frequency of sign changes per frame
- **RMSE Energy**: Loudness and energy variations per frame
- **Chroma STFT**: 12-dimensional pitch content representation
- **Normalization**: Standardization (mean=0, std=1) for all features

### 2. Ensemble Model Architecture

#### CNN Branch
- Input: 3-channel mel-spectrogram (3, 64, T)
- Initial Conv: 7×7 kernels, 64 filters
- Depthwise Separable Blocks:
  - Block 1: 64 → 128 channels
  - Block 2: 128 → 256 channels
  - Block 3: 256 → 512 channels
- Global Average Pooling → 512-dim feature vector
- Advantages: Reduced parameters, efficient computation

#### Bi-LSTM Branch
- Input: Sequence of hand-crafted features (T, feature_dim)
- 2 Bi-LSTM layers with 128 hidden units each
- Bidirectional processing captures temporal context
- Output: 256-dim feature vector (concatenated forward/backward)

#### Fusion & Classification
- Concatenate CNN (512) + LSTM (256) features
- Fusion layer: 768 → 256 with ReLU
- Classification head: 256 → 128 → 8 (emotions)
- Softmax output: Probability distribution over 8 emotions

### 3. Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Loss**: Cross-Entropy Loss
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Regularization**: Dropout (0.5), Batch Normalization
- **Data Split**: Stratified 66:34 train/validation
- **Epochs**: 100 with early stopping

### 4. Evaluation Framework
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PRC
- **Confusion Matrix**: Visualization of misclassifications
- **Cross-Dataset**: TESS, SAVEE, CREMA-D validation
- **Interpretability**: LIME explanations, attention visualization, saliency maps

### 5. Frontend Application
- **Technology**: React.js + Web Audio API
- **Features**:
  - Real-time microphone recording
  - Audio file upload (WAV, MP3, OGG)
  - Emotion display with confidence scores
  - Mel-spectrogram visualization
  - Emotion probability distribution chart
  - Prediction history and export (CSV, JSON)

### 6. REST API
- **POST /api/v1/predict**: Single audio prediction
- **POST /api/v1/predict-stream**: Real-time streaming
- **GET /api/v1/model-info**: Model metadata
- **POST /api/v1/batch-predict**: Batch processing
- **GET /api/v1/health**: Health check
- **Authentication**: API keys, Rate limiting (100 req/min)

### 7. Model Serialization
- **PyTorch (.pth)**: Native format for training/fine-tuning
- **ONNX**: Cross-platform deployment
- **TensorFlow SavedModel**: TensorFlow serving
- **Quantized (INT8)**: 4x size reduction for edge devices

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Accuracy (RAVDESS) | ≥95% | Test set: Actors 20-24 |
| Accuracy (Cross-Dataset) | ≥90% | TESS, SAVEE, CREMA-D |
| Inference Latency | <500ms | Per 3-5 sec audio sample |
| Throughput (GPU) | >10 samples/sec | Batch processing |
| Throughput (CPU) | >1 sample/sec | Single inference |
| Model Size | <20 MB | Full ensemble |
| Quantized Size | <10 MB | INT8 quantization |
| API Response Time | <1 sec | Including preprocessing |
| API Uptime | >99.5% | Production SLA |

## Implementation Phases

### Phase 1: Model Enhancement (Weeks 1-3)
- Implement Bi-LSTM branch
- Implement feature extraction pipeline
- Integrate with existing CNN
- Train ensemble model
- Evaluate on RAVDESS and cross-datasets

### Phase 2: Frontend Development (Weeks 4-6)
- Build React frontend with microphone recording
- Implement file upload and emotion display
- Create visualization components
- Develop backend API server
- Deploy frontend and backend

### Phase 3: Evaluation & Optimization (Weeks 7-9)
- Implement comprehensive evaluation metrics
- Add LIME interpretability
- Implement model quantization and compression
- Performance benchmarking
- Edge device optimization

### Phase 4: Deployment & Documentation (Weeks 10-12)
- Model serialization (PyTorch, ONNX, TensorFlow)
- Docker containerization
- Cloud deployment (AWS/GCP/Azure)
- Comprehensive documentation
- Final testing and launch

## Key Design Decisions

1. **Ensemble Architecture**: CNN captures spatial features from spectrograms, Bi-LSTM captures temporal dependencies in hand-crafted features
2. **Hand-Crafted Features**: Complement learned features, improve interpretability, reduce model complexity
3. **Depthwise Separable Convolutions**: Reduce parameters while maintaining performance for edge deployment
4. **Stratified Sampling**: Ensure balanced emotion distribution in train/validation splits
5. **Feature Normalization**: Standardization ensures equal contribution from different feature types
6. **Learning Rate Scheduling**: Adaptive learning rate improves convergence and prevents overfitting

## Correctness Properties

1. **Feature Extraction Consistency**: Identical audio → identical features
2. **Emotion Classification Validity**: Output is valid probability distribution (sum=1, all in [0,1])
3. **Model Determinism**: Same input → same output (with fixed seed)
4. **Feature Normalization Bounds**: Normalized features within ±3 standard deviations
5. **Ensemble Fusion Correctness**: Fused features preserve information from both branches

## Testing Strategy

- **Unit Tests**: Individual components (>90% coverage)
- **Property-Based Tests**: Hypothesis library for edge cases
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Latency, throughput, memory profiling
- **Cross-Dataset Tests**: Generalization validation
- **Security Tests**: API authentication, input validation

## Deployment Considerations

- **Cloud**: AWS Lambda, Google Cloud Functions, Azure Functions
- **Edge**: Raspberry Pi, Jetson Nano (quantized models)
- **Containerization**: Docker for consistent deployment
- **Monitoring**: Prometheus metrics, ELK logging
- **Security**: HTTPS, API authentication, rate limiting

## Success Criteria

✓ Model accuracy ≥95% on RAVDESS test set  
✓ Model accuracy ≥90% on cross-dataset validation  
✓ Inference latency <500ms on GPU  
✓ Frontend response time <1 second  
✓ API uptime >99.5%  
✓ Code coverage >90%  
✓ All acceptance criteria met  
✓ Comprehensive documentation  
✓ Production deployment successful  

## Files Generated

1. **design.md**: Comprehensive technical design (8000+ words)
   - Architecture overview with Mermaid diagrams
   - Detailed component specifications
   - Data flow diagrams
   - Correctness properties
   - Error handling strategy
   - Testing and performance considerations

2. **requirements.md**: Functional and non-functional requirements
   - 10 functional requirement categories
   - 7 non-functional requirement categories
   - 10 acceptance criteria
   - Data requirements
   - Constraints and assumptions
   - Success metrics
   - Traceability matrix

3. **tasks.md**: Implementation tasks organized by phase
   - 60+ specific implementation tasks
   - 4 implementation phases (12 weeks)
   - Testing tasks
   - Documentation tasks
   - Timeline and resource allocation
   - Risk management

4. **.config.kiro**: Spec configuration file
   - Workflow type: design-first
   - Spec type: feature

## Next Steps

1. **Review Design**: Team review of architecture and design decisions
2. **Approve Requirements**: Stakeholder approval of functional/non-functional requirements
3. **Plan Implementation**: Detailed sprint planning for Phase 1
4. **Setup Development Environment**: Configure development tools and infrastructure
5. **Begin Phase 1**: Start model enhancement implementation

## References

- Existing Codebase: CNN model with 94% accuracy on RAVDESS
- Research Paper: "Speech emotion recognition with lightweight deep neural ensemble model using hand crafted features"
- Dataset: RAVDESS (7356 samples, 8 emotions, 24 actors)
- Cross-Dataset: TESS, SAVEE, CREMA-D

## Contact

**Team**: Asit Jain, Avinash Singh, Prashant Kumar Mishra  
**Institution**: IIT Jodhpur  
**Course**: Speech Understanding (Team X)

---

**Document Version**: 1.0  
**Created**: 2024-01-15  
**Last Updated**: 2024-01-15  
**Status**: Ready for Review
