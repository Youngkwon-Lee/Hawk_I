# HawkEye PD Backend

Python Flask backend for Parkinson's Disease motion analysis

## Features

- ✅ **Movement-based ROI Detection** (100% algorithm-based)
  - Frame differencing
  - Contour detection
  - Multi-contour merging for full-body motion

- ✅ **Auto Task Classification**
  - `finger_tapping` - Small hand ROI (< 2% of frame)
  - `hand_movement` - Larger hand ROI (pronation-supination)
  - `gait` - Full body or large motion area (> 10%)
  - `leg_agility` - Localized foot motion (< 10%)

- ⏳ **MediaPipe Skeleton Extraction** (Coming soon)
- ⏳ **UPDRS Prediction (0-3)** (Coming soon)

## Installation

1. **Create virtual environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Run server**
   ```bash
   python app.py
   ```

Server will start at `http://localhost:5000`

## API Endpoints

### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "dependencies": {
    "opencv": "4.8.1",
    "mediapipe": "0.10.8",
    "pytorch": "2.1.0"
  }
}
```

### 2. Video Analysis
```bash
POST /api/analyze
Content-Type: multipart/form-data

Parameters:
- video_file: File (required)
- patient_id: string (optional)
- test_type: string (optional manual override)
```

Response:
```json
{
  "success": true,
  "video_type": "finger_tapping",  // AUTO-DETECTED!
  "confidence": 0.90,
  "auto_detected": true,
  "roi": {
    "x": 100,
    "y": 200,
    "w": 300,
    "h": 400
  },
  "motion_analysis": {
    "motion_pattern": "localized",
    "motion_area_ratio": 0.015,
    "body_part": "hand"
  },
  "reasoning": "Hand detected with very small ROI (1.5%). Classified as finger_tapping."
}
```

## Project Structure

```
backend/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── routes/
│   ├── health.py          # Health check endpoint
│   └── analyze.py         # Video analysis endpoint
├── services/
│   ├── roi_detector.py    # Movement-based ROI detection
│   ├── task_classifier.py # Auto task classification
│   ├── mediapipe_processor.py  # (Coming soon)
│   └── updrs_model.py     # (Coming soon)
├── models/
│   └── saved_models/      # PyTorch weights (Coming soon)
└── uploads/               # Temporary video storage
```

## Testing

### Test with cURL

```bash
# Finger tapping video
curl -X POST http://localhost:5000/api/analyze \
  -F "video_file=@test_finger_tapping.mp4" \
  -F "patient_id=P001"

# Gait video
curl -X POST http://localhost:5000/api/analyze \
  -F "video_file=@test_gait.mp4"
```

### Test with Python

```python
import requests

url = "http://localhost:5000/api/analyze"
files = {"video_file": open("test_video.mp4", "rb")}
data = {"patient_id": "P001"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Algorithm Details

### Movement-Based ROI Detection

**Step 1: Temporal Frame Differencing**
```python
# Sample 30 frames
# For each consecutive pair:
diff = abs(frame[i] - frame[i-1])
threshold = 10  # pixel intensity
movement_map += diff > threshold
# Normalize to [0, 1]
```

**Step 2: Contour Detection**
```python
# Threshold movement map
binary = movement_map > adaptive_threshold

# Morphological operations
kernel = (5, 5) if threshold < 0.1 else (15, 15)
binary = morphology_close(binary, kernel)

# Find contours
contours = find_contours(binary)

# Merge for full-body (gait)
if len(contours) > 1:
    roi = bounding_rect(merge_all(contours))
else:
    roi = bounding_rect(largest_contour)
```

**Step 3: Body Part Detection**
```python
# MediaPipe Hands detection in ROI
if hands_detected:
    body_part = "hand"

# MediaPipe Pose detection
if full_body_visible:
    body_part = "fullbody"
elif ankles_visible:
    body_part = "foot"
```

### Task Classification

```python
# Hand + Small ROI → Finger Tapping
if body_part == "hand" and roi_size < 2%:
    task_type = "finger_tapping"

# Hand + Large ROI → Hand Movement
elif body_part == "hand" and roi_size >= 2%:
    task_type = "hand_movement"

# Foot + Large Motion → Gait
elif body_part == "foot" and motion_area > 10%:
    task_type = "gait"

# Foot + Small Motion → Leg Agility
elif body_part == "foot" and motion_area < 10%:
    task_type = "leg_agility"
```

## Performance

- Movement map calculation: ~800ms (30 frames)
- ROI detection: ~50ms
- Body part detection: ~400ms (5 frames)
- **Total: ~1.3s per video**

## License

MIT

## Contributors

- HawkEye PD Team
- FastEval Parkinsonism (Reference Implementation)
