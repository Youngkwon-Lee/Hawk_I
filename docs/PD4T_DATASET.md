# PD4T Dataset Structure

## Overview
PD4T (Parkinson's Disease 4-Task) dataset for MDS-UPDRS validation.

## Location
```
C:/Users/YK/tulip/PD4T/PD4T/PD4T/
```

## Directory Structure
```
PD4T/
├── README.txt
├── Annotations/
│   ├── Finger tapping/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── Gait/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── Hand movement/
│   │   ├── train.csv
│   │   └── test.csv
│   └── Leg agility/
│       ├── train.csv
│       └── test.csv
└── Videos/
    ├── Finger tapping/
    │   ├── l/          # Left hand videos
    │   │   └── *.mp4
    │   └── r/          # Right hand videos
    │       └── *.mp4
    ├── Gait/
    │   ├── 001/        # Subject folders
    │   ├── 003/
    │   ├── 004/
    │   └── .../
    │       └── *.mp4   # Video files (e.g., 12-104704.mp4)
    ├── Hand movement/
    │   └── .../
    └── Leg agility/
        └── .../
```

## Annotation CSV Format
**No header row** - use `header=None` when loading with pandas

| Column | Name | Description |
|--------|------|-------------|
| 0 | video | Video identifier (e.g., "15-001760_009") |
| 1 | frames | Number of frames in video |
| 2 | score | MDS-UPDRS score (0-4) |

### Loading Example
```python
import pandas as pd

# Correct way to load annotations
df = pd.read_csv('Annotations/Gait/train.csv',
                 header=None,
                 names=['video', 'frames', 'score'])
```

## Score Distribution

### Finger Tapping (train.csv)
| Score | Count |
|-------|-------|
| 0 | 102 |
| 1 | 347 |
| 2 | 126 |
| 3 | 13 |
| 4 | 2 |

### Gait (train.csv)
| Score | Count |
|-------|-------|
| 0 | 152 |
| 1 | 114 |
| 2 | 41 |
| 3 | 3 |

## Video Naming Convention
- Format: `YY-XXXXXX.mp4` (e.g., `12-104704.mp4`)
- For Finger Tapping with laterality: `YY-XXXXXX_l.mp4` (left) or `YY-XXXXXX_r.mp4` (right)
- Annotation video ID format: `YY-XXXXXX_NNN` where NNN is a sequence number

## Finding Videos

### Finger Tapping
```python
# Annotation: "15-005087_l" -> Video: "Videos/Finger tapping/l/15-005087.mp4"
video_id = "15-005087_l"
parts = video_id.rsplit('_', 1)  # ['15-005087', 'l']
video_path = f"Videos/Finger tapping/{parts[1]}/{parts[0]}.mp4"
```

### Gait
```python
# Annotation: "14-005690_026" -> Find in subdirectories
video_id = "14-005690_026"
base_name = video_id.rsplit('_', 1)[0]  # "14-005690"
# Search in Videos/Gait/*/14-005690.mp4
```

## Sample Videos for Testing

### Finger Tapping
| Score | Video ID | Path |
|-------|----------|------|
| 1 | 14-005717_l | Videos/Finger tapping/l/14-005717.mp4 |
| 2 | 15-003071_r | Videos/Finger tapping/r/15-003071.mp4 |
| 3 | 15-005087_l | Videos/Finger tapping/l/15-005087.mp4 |

### Gait
| Score | Video ID | Path Pattern |
|-------|----------|--------------|
| 1 | 14-005690_026 | Videos/Gait/*/14-005690.mp4 |
| 2 | 15-003012_024 | Videos/Gait/*/15-003012.mp4 |
| 3 | 13-007586_004 | Videos/Gait/*/13-007586.mp4 |

## Usage Notes
1. Videos are recorded at 30 FPS
2. Finger tapping videos show hand close-up
3. Gait videos show full body walking
4. World landmarks from MediaPipe provide real meter measurements for gait

---

# Feature Extraction Report (2024-12-04)

## Extracted Data Summary

### Finger Tapping Features

#### Version 1 (Clinical Only) - 10 features
| Feature | Description |
|---------|-------------|
| finger_distance | Thumb tip to index tip distance |
| dist_velocity | Rate of change of finger distance |
| dist_accel | Acceleration of finger distance |
| thumb_speed | Frame-to-frame thumb displacement |
| index_speed | Frame-to-frame index finger displacement |
| combined_speed | Sum of thumb and index speeds |
| thumb_from_wrist | Distance from thumb to wrist |
| index_from_wrist | Distance from index finger to wrist |
| normalized_distance | Finger distance / hand size |
| hand_size | Max(thumb_from_wrist, index_from_wrist) |

#### Version 2 (Raw 3D + Clinical) - 73 features
| Category | Features | Count |
|----------|----------|-------|
| Raw 3D Coordinates | 21 hand landmarks × 3 (x, y, z) | 63 |
| Clinical Features | Same as v1 | 10 |
| **Total** | | **73** |

**Hand Landmarks (MediaPipe Hands):**
- WRIST (0), THUMB_CMC (1), THUMB_MCP (2), THUMB_IP (3), THUMB_TIP (4)
- INDEX_MCP (5), INDEX_PIP (6), INDEX_DIP (7), INDEX_TIP (8)
- MIDDLE_MCP (9), MIDDLE_PIP (10), MIDDLE_DIP (11), MIDDLE_TIP (12)
- RING_MCP (13), RING_PIP (14), RING_DIP (15), RING_TIP (16)
- PINKY_MCP (17), PINKY_PIP (18), PINKY_DIP (19), PINKY_TIP (20)

### Gait Features

#### Version 1 (Clinical Only) - 30 features
| Category | Features | Count |
|----------|----------|-------|
| Position Features | step_width, hip_height, trunk_angle, left_knee_angle, right_knee_angle, left_ankle_height, right_ankle_height, left_arm_swing, right_arm_swing, body_sway, stride_proxy, hip_asymmetry, shoulder_asymmetry, left_hip_angle, right_hip_angle | 15 |
| Velocity Features | Same features with _vel suffix | 15 |
| **Total** | | **30** |

#### Version 2 (Raw 3D + Clinical) - 129 features
| Category | Features | Count |
|----------|----------|-------|
| Raw 3D Coordinates | 33 pose landmarks × 3 (x, y, z) | 99 |
| Clinical Position | Same as v1 position | 15 |
| Clinical Velocity | Same as v1 velocity | 15 |
| **Total** | | **129** |

**Pose Landmarks (MediaPipe Pose):**
- Face: nose (0), eyes (1-6), ears (7-8), mouth (9-10)
- Upper Body: shoulders (11-12), elbows (13-14), wrists (15-16)
- Hands: pinky (17-18), index (19-20), thumb (21-22)
- Lower Body: hips (23-24), knees (25-26), ankles (27-28)
- Feet: heels (29-30), foot_index (31-32)

## Extracted Data Files

### HPC Data Directory: `hpc/data/`

| File | Task | Version | Samples | Shape | Size |
|------|------|---------|---------|-------|------|
| `finger_train_v2.pkl` | Finger Tapping | v2 | 568 | (568, 150, 73) | 50MB |
| `finger_valid_v2.pkl` | Finger Tapping | v2 | 103 | (103, 150, 73) | 9MB |
| `finger_test_v2.pkl` | Finger Tapping | v2 | 135 | (135, 150, 73) | 12MB |
| `gait_train_data.pkl` | Gait | v1 | 353 | (353, 300, 30) | 25MB |
| `gait_test_data.pkl` | Gait | v1 | 73 | (73, 300, 30) | 5MB |

### Data Structure (pkl format)
```python
{
    'X': np.array,        # Features: (samples, seq_len, features)
    'y': np.array,        # Labels: (samples,) - UPDRS scores 0-4
    'ids': list,          # Video IDs
    'task': str,          # 'finger_tapping' or 'gait'
    'version': str,       # 'v1' or 'v2_raw3d_clinical'
    'features': list,     # Feature names
    'feature_groups': dict  # Feature index ranges
}
```

## Label Distribution

### Finger Tapping (Annotations_split)
| Split | Total | UPDRS 0 | UPDRS 1 | UPDRS 2 | UPDRS 3 | UPDRS 4 |
|-------|-------|---------|---------|---------|---------|---------|
| Train | 568 | 89 (15.7%) | 290 (51.1%) | 163 (28.7%) | 24 (4.2%) | 2 (0.4%) |
| Valid | 103 | 13 (12.6%) | 57 (55.3%) | 26 (25.2%) | 7 (6.8%) | 0 (0%) |
| Test | 135 | 25 (18.5%) | 57 (42.2%) | 39 (28.9%) | 12 (8.9%) | 2 (1.5%) |

### Gait (Annotations_split)
| Split | Total | UPDRS 0 | UPDRS 1 | UPDRS 2 | UPDRS 3 |
|-------|-------|---------|---------|---------|---------|
| Train | 296 | 134 (45.3%) | 107 (36.1%) | 47 (15.9%) | 8 (2.7%) |
| Valid | 57 | 29 (50.9%) | 19 (33.3%) | 9 (15.8%) | 0 (0%) |
| Test | 73 | 34 (46.6%) | 28 (38.4%) | 10 (13.7%) | 1 (1.4%) |

## Prior Research Comparison

| Study | Task | Features | Accuracy |
|-------|------|----------|----------|
| Li et al. (2022) | Finger Tapping | 3D spatial-temporal | 81.2% |
| PD4T Benchmark | Finger Tapping | Video-based | ~60-70% |
| **Our v1** | Finger Tapping | 10 clinical | 62% |
| **Our v2** | Finger Tapping | 73 (raw 3D + clinical) | **Target: 80%+** |

## HPC Training Commands

### Gait v1 Training
```bash
ssh gun3856@hpc.cau.ac.kr
cd /home2/gun3856/hpc
conda activate triage
python scripts/train_gait_gpu.py
```

### Finger v2 Training
```bash
python scripts/train_lstm_v2.py --task finger
```

### Model Architectures
| Model | Description | Parameters |
|-------|-------------|------------|
| AttentionLSTM | Bidirectional LSTM + Attention | ~500K |
| TransformerModel | Transformer Encoder | ~1M |
| SpatialTemporalLSTM | Spatial + Temporal encoders | ~600K |

## Data Transfer Commands

```bash
# Scripts
scp hpc/scripts/train_gait_gpu.py gun3856@hpc.cau.ac.kr:/home2/gun3856/hpc/scripts/
scp hpc/scripts/train_lstm_v2.py gun3856@hpc.cau.ac.kr:/home2/gun3856/hpc/scripts/

# Gait v1 Data
scp hpc/data/gait_*.pkl gun3856@hpc.cau.ac.kr:/home2/gun3856/hpc/data/

# Finger v2 Data
scp hpc/data/finger_*_v2.pkl gun3856@hpc.cau.ac.kr:/home2/gun3856/hpc/data/
```
