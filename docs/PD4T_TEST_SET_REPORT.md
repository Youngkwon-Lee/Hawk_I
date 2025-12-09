# PD4T Test Set Detailed Analysis Report

**Report Generated**: 2025-11-24
**Dataset**: PD4T (Parkinson's Disease 4 Tasks)
**Analysis Scope**: Test Set Only

---

## Executive Summary

The PD4T test set contains **783 videos** from **19 unique patients** performing 4 different motor assessment tasks. The dataset exhibits significant class imbalance with normal/slight impairment (scores 0-1) representing 81.6% of all samples, while severe impairment (scores 3-4) accounts for only 2.9% of cases.

---

## 1. Overall Test Set Statistics

### Basic Metrics
- **Total test videos**: 783
- **Unique patients (subjects)**: 19
- **Tasks**: 4 (Gait, Finger tapping, Hand movement, Leg agility)
- **Average videos per patient**: 41.2
- **Patient-level split**: Ensures no data leakage between train/test

### Patient IDs in Test Set
```
001, 005, 006, 007, 008, 009, 011, 012, 013, 019,
022, 023, 027, 037, 038, 039, 042, 044, 057
```

### Aggregate Score Distribution (All Tasks)

| Score | Videos | Percentage | Severity Level |
|-------|--------|------------|----------------|
| **0** | 275 | 35.1% | Normal (No problems) |
| **1** | 364 | 46.5% | Slight impairment |
| **2** | 121 | 15.5% | Mild impairment |
| **3** | 22 | 2.8% | Moderate impairment |
| **4** | 1 | 0.1% | Severe impairment |

**Key Observation**:
- Scores 0-1 dominate: 81.6% of all samples
- Severe cases (3-4): Only 2.9% - significant class imbalance

---

## 2. Task-wise Analysis

### 2.1 Gait Task

**Test Statistics**:
- Videos: 116
- Unique subjects: 8
- Frame range: 325 - 10,688 frames
- Mean frames: 1005.7
- Median frames: 826.0

**Subject Distribution**:
| Subject ID | Videos | Videos/Subject |
|------------|--------|----------------|
| 019 | 15 | 12.9% |
| 022 | 16 | 13.8% |
| 023 | 14 | 12.1% |
| 037 | 14 | 12.1% |
| 038 | 14 | 12.1% |
| 039 | 14 | 12.1% |
| 042 | 13 | 11.2% |
| 044 | 16 | 13.8% |

**Score Distribution**:
| Score | Videos | Percentage |
|-------|--------|------------|
| 0 | 44 | 37.9% |
| 1 | 44 | 37.9% |
| 2 | 23 | 19.8% |
| 3 | 5 | 4.3% |

**Observations**:
- Longest videos in dataset (avg ~1006 frames)
- Balanced distribution between normal (0) and slight (1)
- No score 4 cases in test set

---

### 2.2 Finger Tapping Task

**Test Statistics**:
- Videos: 216
- Unique subjects: 8
- Frame range: 131 - 550 frames
- Mean frames: 200.1
- Median frames: 172.0

**Subject Distribution**:
| Subject ID | Videos | Videos/Subject |
|------------|--------|----------------|
| 006 | 28 | 13.0% |
| 009 | 28 | 13.0% |
| 011 | 22 | 10.2% |
| 012 | 27 | 12.5% |
| 013 | 23 | 10.6% |
| 022 | 32 | 14.8% |
| 027 | 28 | 13.0% |
| 057 | 28 | 13.0% |

**Score Distribution**:
| Score | Videos | Percentage |
|-------|--------|------------|
| 0 | 50 | 23.1% |
| 1 | 118 | 54.6% |
| 2 | 38 | 17.6% |
| 3 | 10 | 4.6% |

**Observations**:
- Highest number of test videos (216)
- Score 1 (slight) is dominant (54.6%)
- Relatively short videos (avg ~200 frames)

---

### 2.3 Hand Movement Task

**Test Statistics**:
- Videos: 227
- Unique subjects: 8
- Frame range: 133 - 801 frames
- Mean frames: 226.8
- Median frames: 200.0

**Subject Distribution**:
| Subject ID | Videos | Videos/Subject |
|------------|--------|----------------|
| 006 | 28 | 12.3% |
| 009 | 28 | 12.3% |
| 011 | 28 | 12.3% |
| 012 | 28 | 12.3% |
| 013 | 27 | 11.9% |
| 022 | 32 | 14.1% |
| 027 | 28 | 12.3% |
| 057 | 28 | 12.3% |

**Score Distribution**:
| Score | Videos | Percentage |
|-------|--------|------------|
| 0 | 67 | 29.5% |
| 1 | 108 | 47.6% |
| 2 | 48 | 21.1% |
| 3 | 3 | 1.3% |
| 4 | 1 | 0.4% |

**Observations**:
- Most videos in test set (227)
- Contains the ONLY score 4 case in entire test set
- Relatively even distribution across subjects

---

### 2.4 Leg Agility Task

**Test Statistics**:
- Videos: 224
- Unique subjects: 8
- Frame range: 133 - 686 frames
- Mean frames: 214.4
- Median frames: 205.5

**Subject Distribution**:
| Subject ID | Videos | Videos/Subject |
|------------|--------|----------------|
| 001 | 28 | 12.5% |
| 005 | 28 | 12.5% |
| 006 | 28 | 12.5% |
| 007 | 28 | 12.5% |
| 008 | 28 | 12.5% |
| 009 | 28 | 12.5% |
| 011 | 28 | 12.5% |
| 042 | 28 | 12.5% |

**Score Distribution**:
| Score | Videos | Percentage |
|-------|--------|------------|
| 0 | 114 | 50.9% |
| 1 | 94 | 42.0% |
| 2 | 12 | 5.4% |
| 3 | 4 | 1.8% |

**Observations**:
- Most balanced subject distribution (exactly 28 videos each)
- Highest proportion of normal cases (50.9%)
- Fewest moderate/severe cases (1.8% score 3)

---

## 3. Cross-Task Subject Analysis

### 3.1 Subject Task Participation

| Subject ID | Tasks | Task Names |
|------------|-------|------------|
| 001 | 1 | Leg agility |
| 005 | 1 | Leg agility |
| 006 | 3 | Finger tapping, Hand movement, Leg agility |
| 007 | 1 | Leg agility |
| 008 | 1 | Leg agility |
| 009 | 3 | Finger tapping, Hand movement, Leg agility |
| 011 | 3 | Finger tapping, Hand movement, Leg agility |
| 012 | 2 | Finger tapping, Hand movement |
| 013 | 2 | Finger tapping, Hand movement |
| 019 | 1 | Gait |
| 022 | 3 | Gait, Finger tapping, Hand movement |
| 023 | 1 | Gait |
| 027 | 2 | Finger tapping, Hand movement |
| 037 | 1 | Gait |
| 038 | 1 | Gait |
| 039 | 1 | Gait |
| 042 | 2 | Gait, Leg agility |
| 044 | 1 | Gait |
| 057 | 2 | Finger tapping, Hand movement |

### 3.2 Multi-Task Subjects

**3 Tasks (4 subjects)**:
- 006, 009, 011: Finger tapping + Hand movement + Leg agility
- 022: Gait + Finger tapping + Hand movement

**2 Tasks (5 subjects)**:
- 012, 013, 027, 057: Finger tapping + Hand movement
- 042: Gait + Leg agility

**Single Task (10 subjects)**:
- Gait only: 019, 023, 037, 038, 039, 044
- Leg agility only: 001, 005, 007, 008

---

## 4. Evaluation Considerations

### 4.1 Class Imbalance Strategies

Given the severe class imbalance (81.6% for scores 0-1):

**Recommended Approaches**:
1. **Weighted Loss Functions**: Inverse frequency weighting
   - Score 0: weight ~0.36
   - Score 1: weight ~0.27
   - Score 2: weight ~0.82
   - Score 3: weight ~4.5
   - Score 4: weight ~78.3

2. **Ordinal Regression**: Exploit natural ordering of scores
   - Use cumulative link models
   - Minimize absolute error instead of classification error

3. **Evaluation Metrics**:
   - Primary: **Weighted Kappa** (accounts for ordinal nature)
   - Secondary: **MAE** (Mean Absolute Error)
   - Avoid: Simple accuracy (misleading with imbalance)

### 4.2 Temporal Sampling Strategies

Frame counts vary significantly across tasks:

| Task | Min | Max | Mean | Strategy |
|------|-----|-----|------|----------|
| Gait | 325 | 10,688 | 1006 | Sparse sampling (fps=4.0) |
| Finger tapping | 131 | 550 | 200 | Dense sampling (fps=25.0) |
| Hand movement | 133 | 801 | 227 | Dense sampling (fps=25.0) |
| Leg agility | 133 | 686 | 214 | Medium sampling (fps=20.0) |

**Qwen Model Configuration** (from reference code):
```python
video_config = {
    "Finger tapping": {"fps": 25.0, "max_frames": 150},
    "Hand movement": {"fps": 25.0, "max_frames": 150},
    "Leg agility": {"fps": 20.0, "max_frames": 120},
    "Gait": {"fps": 4.0, "max_frames": 80}
}
```

### 4.3 Expected Performance Ranges

Based on similar studies and class imbalance:

| Metric | Conservative | Optimistic | Notes |
|--------|-------------|------------|-------|
| Accuracy | 0.50-0.60 | 0.65-0.75 | Majority class baseline: 0.465 |
| MAE | 0.4-0.6 | 0.2-0.4 | Lower is better |
| Weighted Kappa | 0.40-0.55 | 0.60-0.75 | >0.60 is substantial agreement |

---

## 5. VLM Evaluation Setup

### 5.1 GPT-4o Configuration

**Frame Extraction**:
```python
frame_config = {
    "Gait": {"max_frames": 16},
    "Finger tapping": {"max_frames": 12},
    "Hand movement": {"max_frames": 12},
    "Leg agility": {"max_frames": 12}
}
```

**API Parameters**:
- Model: `gpt-4o`
- Temperature: 0.0 (deterministic)
- Max tokens: 300
- Frame resolution: 512x512

**Cost Estimate**:
- Per sample: ~$0.08-0.15
- Total test set (783): ~$62-117

### 5.2 Gemini 2.0 Flash Configuration

**Video Upload**:
- Direct video file upload (no frame extraction)
- Automatic processing and cleanup
- Rate limiting: 1 second between requests

**API Parameters**:
- Model: `gemini-2.0-flash-exp`
- Temperature: 0.0 (deterministic)
- Max output tokens: 300

**Cost Estimate**:
- Per sample: $0 (free tier)
- Total test set (783): $0

---

## 6. Key Findings and Recommendations

### 6.1 Dataset Characteristics

1. **Class Imbalance**: Critical challenge requiring weighted approaches
2. **Temporal Variation**: 30x difference in video length (325-10,688 frames)
3. **Patient Diversity**: 19 subjects with varying task participation
4. **Realistic Split**: Patient-level split prevents data leakage

### 6.2 Evaluation Best Practices

1. **Use Weighted Kappa**: Primary metric for ordinal data with imbalance
2. **Report Per-Task Performance**: Tasks show different characteristics
3. **Analyze Confusion Matrix**: Understand systematic errors
4. **Consider Clinical Relevance**: 1-point error is less severe than 3-point error

### 6.3 Future Improvements

1. **Ensemble Methods**: Combine GPT-4o and Gemini predictions
2. **Temporal Modeling**: Exploit full video context (not just frames)
3. **Multi-Task Learning**: Leverage shared representations across tasks
4. **Active Learning**: Focus on hard cases (scores 3-4)

---

## 7. Data Quality Notes

### 7.1 Verified Aspects
- All videos exist in expected directory structure
- CSV annotations properly formatted
- Patient IDs consistent across tasks
- Score distributions align with clinical reality

### 7.2 Potential Issues
- Single score 4 case (potential label noise or true rare event)
- Large frame count variation in Gait (325-10,688)
- Some subjects only in single task

---

## Appendix: File Paths

### Test Set Annotations
```
PD4T/PD4T/PD4T/Annotations/
├── Gait/test.csv (116 samples)
├── Finger tapping/test.csv (216 samples)
├── Hand movement/test.csv (227 samples)
└── Leg agility/test.csv (224 samples)
```

### Video Files
```
PD4T/PD4T/PD4T/Videos/
├── Gait/{patient_id}/{visit_number}.mp4
├── Finger tapping/{patient_id}/{visit_number}_{l|r}.mp4
├── Hand movement/{patient_id}/{visit_number}_{l|r}.mp4
└── Leg agility/{patient_id}/{visit_number}_{l|r}.mp4
```

---

**Contact**: a.dadashzadeh@bristol.ac.uk (PD4T Dataset)
**Report Version**: 1.0
**Last Updated**: 2025-11-24
