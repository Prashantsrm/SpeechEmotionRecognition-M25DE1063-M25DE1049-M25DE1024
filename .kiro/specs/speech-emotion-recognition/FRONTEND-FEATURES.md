# Frontend Features: Voice Input & File Upload
## Real-Time Emotion Detection Interface

---

## Overview

The frontend provides two ways to input audio for emotion detection:
1. **Direct Voice Recording** - Real-time microphone input
2. **Manual File Upload** - Upload pre-recorded audio files

---

## Feature 1: Direct Voice Recording

### How It Works

```
User clicks "Start Recording" 
    ↓
Microphone permission requested
    ↓
Audio captured in real-time
    ↓
Visual waveform displayed
    ↓
User clicks "Stop Recording"
    ↓
Audio sent to backend
    ↓
Emotion prediction displayed
```

### User Interface

```
┌─────────────────────────────────────┐
│  Speech Emotion Recognition         │
├─────────────────────────────────────┤
│                                     │
│  🎤 MICROPHONE RECORDING            │
│                                     │
│  [Start Recording] [Stop Recording] │
│                                     │
│  Recording Duration: 0:05           │
│  Audio Level: ████████░░ 80%        │
│                                     │
│  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁ (Waveform)        │
│                                     │
│  [Analyze Emotion]                  │
│                                     │
└─────────────────────────────────────┘
```

### Technical Implementation

```python
# Backend: Receive audio stream
@app.route('/api/v1/predict-stream', methods=['POST'])
def predict_stream():
    """Handle real-time audio streaming"""
    
    # Receive audio chunks
    audio_chunks = []
    
    while True:
        chunk = request.get_data()
        if not chunk:
            break
        audio_chunks.append(chunk)
    
    # Combine chunks
    audio_data = b''.join(audio_chunks)
    
    # Convert to numpy array
    y = np.frombuffer(audio_data, dtype=np.float32)
    
    # Extract features and predict
    features = extractor.extract_all_features(y)
    prediction = model.predict(features)
    
    return jsonify(prediction)
```

### Features

✅ **Real-time Waveform Visualization**
- Shows audio amplitude over time
- Updates as user speaks
- Visual feedback of recording quality

✅ **Audio Level Indicator**
- Shows microphone input level
- Helps user maintain consistent volume
- Prevents too quiet/loud recordings

✅ **Recording Duration Display**
- Shows elapsed time
- Helps user know when to stop
- Typical duration: 3-5 seconds

✅ **Microphone Permission Handling**
- Requests permission on first use
- Clear error messages if denied
- Works across browsers

✅ **Noise Detection**
- Warns if background noise is too high
- Suggests quiet environment
- Improves prediction accuracy

---

## Feature 2: Manual File Upload

### How It Works

```
User selects "Upload File"
    ↓
File browser opens
    ↓
User selects audio file (WAV, MP3, OGG)
    ↓
File validation (format, size, duration)
    ↓
Upload progress shown
    ↓
File sent to backend
    ↓
Emotion prediction displayed
```

### Supported Formats

| Format | Extension | Sample Rate | Notes |
|--------|-----------|-------------|-------|
| WAV | .wav | 16-48 kHz | Lossless, recommended |
| MP3 | .mp3 | 16-48 kHz | Compressed, common |
| OGG | .ogg | 16-48 kHz | Open format |
| FLAC | .flac | 16-48 kHz | Lossless |

### File Constraints

- **Maximum Size**: 50 MB
- **Minimum Duration**: 1 second
- **Maximum Duration**: 30 seconds
- **Recommended Duration**: 3-5 seconds

### User Interface

```
┌─────────────────────────────────────┐
│  Speech Emotion Recognition         │
├─────────────────────────────────────┤
│                                     │
│  📁 FILE UPLOAD                     │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Drag & drop audio file here │   │
│  │ or click to browse          │   │
│  └─────────────────────────────┘   │
│                                     │
│  Supported: WAV, MP3, OGG, FLAC    │
│  Max size: 50 MB                   │
│                                     │
│  [Browse Files]                     │
│                                     │
│  Selected: sample_audio.wav         │
│  Size: 2.5 MB                       │
│  Duration: 4.2 seconds             │
│                                     │
│  [Upload & Analyze]                │
│                                     │
│  Upload Progress: ████████░░ 80%   │
│                                     │
└─────────────────────────────────────┘
```

### Technical Implementation

```python
# Backend: Handle file upload
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    
    # Validate file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Validate format
    allowed_formats = {'wav', 'mp3', 'ogg', 'flac'}
    file_ext = audio_file.filename.rsplit('.', 1)[1].lower()
    
    if file_ext not in allowed_formats:
        return jsonify({'error': f'Format {file_ext} not supported'}), 400
    
    # Validate size
    audio_file.seek(0, 2)  # Seek to end
    file_size = audio_file.tell()
    audio_file.seek(0)  # Seek back to start
    
    if file_size > 50 * 1024 * 1024:  # 50 MB
        return jsonify({'error': 'File too large'}), 400
    
    # Load and process audio
    try:
        audio_data = io.BytesIO(audio_file.read())
        y, sr = librosa.load(audio_data, sr=22050)
        
        # Validate duration
        duration = len(y) / sr
        if duration < 1:
            return jsonify({'error': 'Audio too short (min 1 second)'}), 400
        if duration > 30:
            return jsonify({'error': 'Audio too long (max 30 seconds)'}), 400
        
        # Extract features and predict
        features = extractor.extract_all_features(y)
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        
        return jsonify({
            'emotion': emotions[torch.argmax(probabilities).item()],
            'confidence': float(probabilities.max()),
            'probabilities': probabilities.tolist(),
            'duration': duration
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Features

✅ **Drag & Drop Upload**
- Drag audio file directly onto interface
- No need to browse file system
- Intuitive user experience

✅ **File Validation**
- Checks format before upload
- Validates file size
- Checks audio duration
- Clear error messages

✅ **Upload Progress**
- Shows upload percentage
- Estimated time remaining
- Cancel upload option

✅ **File Preview**
- Shows filename
- Displays file size
- Shows audio duration
- Allows user to confirm before analysis

✅ **Multiple File Support**
- Upload multiple files sequentially
- Batch processing option
- Download results as CSV

---

## Unified Interface

### Combined View

```
┌──────────────────────────────────────────────┐
│  Speech Emotion Recognition System           │
├──────────────────────────────────────────────┤
│                                              │
│  [🎤 Record] [📁 Upload] [📊 History]       │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ RECORD AUDIO                         │   │
│  │                                      │   │
│  │ [Start Recording] [Stop Recording]   │   │
│  │ Duration: 0:05 | Level: ████░░ 80%  │   │
│  │ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁ (Waveform)          │   │
│  │                                      │   │
│  │ [Analyze]                            │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ UPLOAD FILE                          │   │
│  │                                      │   │
│  │ Drag & drop or [Browse Files]        │   │
│  │ Supported: WAV, MP3, OGG, FLAC       │   │
│  │                                      │   │
│  │ Selected: sample.wav (2.5 MB, 4.2s) │   │
│  │ [Upload & Analyze]                   │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ EMOTION PREDICTION                   │   │
│  │                                      │   │
│  │ 😊 HAPPY (87% confidence)            │   │
│  │                                      │   │
│  │ Neutral:    ░░░░░░░░░░ 5%            │   │
│  │ Calm:       ░░░░░░░░░░ 3%            │   │
│  │ Happy:      ████████░░ 87%           │   │
│  │ Sad:        ░░░░░░░░░░ 2%            │   │
│  │ Angry:      ░░░░░░░░░░ 1%            │   │
│  │ Fearful:    ░░░░░░░░░░ 1%            │   │
│  │ Disgust:    ░░░░░░░░░░ 1%            │   │
│  │ Surprised:  ░░░░░░░░░░ 0%            │   │
│  │                                      │   │
│  │ [Export Results] [Save to History]   │   │
│  └──────────────────────────────────────┘   │
│                                              │
└──────────────────────────────────────────────┘
```

---

## Emotion Display

### Visual Representation

```
Emotion: HAPPY
Confidence: 87%

Visual Indicator:
┌─────────────────────────────────────┐
│ 😊 HAPPY                            │
│ ████████████████████░░░░░░░░░░░░░░░ │
│ 87%                                 │
└─────────────────────────────────────┘

Color Coding:
- Neutral:   Gray (#808080)
- Calm:      Light Blue (#87CEEB)
- Happy:     Gold (#FFD700)
- Sad:       Royal Blue (#4169E1)
- Angry:     Orange Red (#FF4500)
- Fearful:   Purple (#9932CC)
- Disgust:   Forest Green (#228B22)
- Surprised: Hot Pink (#FF69B4)
```

### Confidence Score Interpretation

```
Confidence Range | Interpretation | Recommendation
─────────────────┼────────────────┼──────────────────
90-100%          | Very High      | Trust prediction
80-90%           | High           | Trust prediction
70-80%           | Medium         | Generally reliable
60-70%           | Low            | May be uncertain
<60%             | Very Low       | Uncertain, retry
```

---

## Visualization Features

### Mel-Spectrogram Display

```
Shows frequency content over time:

Frequency (Hz)
▲
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
└─────────────────────────────────────► Time (seconds)

Darker = More energy at that frequency/time
```

### Waveform Display

```
Shows audio amplitude over time:

Amplitude
▲
│     ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
│   ▁▃▅▇█████████▇▅▃▁
│ ▁▃▅▇█████████████▇▅▃▁
│▁▃▅▇█████████████████▇▅▃▁
└─────────────────────────────────────► Time (seconds)
```

---

## Export & History

### Export Options

```
Export Formats:
- CSV: emotion, confidence, probabilities, timestamp
- JSON: Full prediction data with metadata
- PDF: Report with visualizations
- PNG: Emotion chart image
```

### Prediction History

```
┌─────────────────────────────────────────────┐
│ PREDICTION HISTORY                          │
├─────────────────────────────────────────────┤
│ Time      | Audio      | Emotion | Conf.   │
├─────────────────────────────────────────────┤
│ 14:32:15  | recording1 | Happy   | 87%     │
│ 14:28:42  | sample.wav | Sad     | 92%     │
│ 14:25:10  | recording2 | Angry   | 78%     │
│ 14:20:33  | file.mp3   | Calm    | 85%     │
└─────────────────────────────────────────────┘

Actions: [View] [Export] [Delete] [Share]
```

---

## Mobile Responsiveness

### Mobile View

```
┌──────────────────┐
│ Emotion Detector │
├──────────────────┤
│                  │
│ [🎤] [📁]        │
│                  │
│ ┌──────────────┐ │
│ │ Start Record │ │
│ └──────────────┘ │
│                  │
│ ┌──────────────┐ │
│ │ Upload File  │ │
│ └──────────────┘ │
│                  │
│ ┌──────────────┐ │
│ │ 😊 HAPPY     │ │
│ │ 87%          │ │
│ └──────────────┘ │
│                  │
│ ████████░░ 87%   │
│                  │
└──────────────────┘
```

---

## Accessibility Features

✅ **Keyboard Navigation**
- Tab through all controls
- Enter to activate buttons
- Space to start/stop recording

✅ **Screen Reader Support**
- ARIA labels for all elements
- Descriptive button text
- Emotion descriptions

✅ **Color Contrast**
- WCAG AA compliant
- High contrast mode option
- Color-blind friendly palette

✅ **Text Alternatives**
- Alt text for all images
- Transcripts for audio
- Descriptions for visualizations

---

## Error Handling

### Common Errors & Solutions

```
Error: "Microphone permission denied"
Solution: Allow microphone access in browser settings

Error: "Audio file format not supported"
Solution: Convert to WAV, MP3, OGG, or FLAC

Error: "Audio too short (min 1 second)"
Solution: Record or upload longer audio (3-5 seconds recommended)

Error: "Audio too long (max 30 seconds)"
Solution: Use shorter audio clip

Error: "Upload failed"
Solution: Check internet connection, try again

Error: "Prediction failed"
Solution: Try different audio, check backend status
```

---

## Performance Optimization

### Frontend Optimization
- Lazy load components
- Cache predictions
- Compress audio before upload
- Use Web Workers for processing

### Backend Optimization
- Model quantization for faster inference
- Batch processing for multiple files
- Caching for repeated predictions
- Load balancing for concurrent requests

---

## Security Considerations

✅ **Audio Privacy**
- Audio not stored permanently
- Temporary files deleted after processing
- HTTPS encryption for all transfers
- User consent for data collection

✅ **Input Validation**
- File format validation
- File size limits
- Audio duration limits
- Malware scanning

✅ **API Security**
- Rate limiting (100 requests/minute)
- API key authentication
- CORS configuration
- Input sanitization

---

## Future Enhancements

🔮 **Planned Features**
- Real-time emotion tracking over time
- Emotion intensity visualization
- Speaker identification
- Multi-speaker emotion detection
- Emotion transition analysis
- Comparison with baseline emotions
- Emotion prediction confidence intervals
- Batch processing dashboard

---

**Document Version**: 1.0  
**Created**: 2024-01-15  
**Status**: Complete
