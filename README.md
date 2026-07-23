# 🦅 Hawkeye - Parkinson's Disease Motor Assessment

AI-powered motor function assessment system for Parkinson's Disease using video analysis.

## 📋 Overview

Hawkeye analyzes patient movement videos to provide objective MDS-UPDRS motor assessments using:
- **MediaPipe** pose estimation
- **LSTM/ML models** for temporal pattern analysis
- **Multi-agent AI system** for clinical interpretation

### Supported Tasks (MDS-UPDRS Part III)
| Task | Description | Score Range |
|------|-------------|-------------|
| Gait | Walking pattern analysis | 0-4 |
| Finger Tapping | Repetitive finger movements | 0-4 |
| Hand Movement | Hand opening/closing | 0-4 |
| Leg Agility | Leg lifting movements | 0-4 |

## 📁 Project Structure

```
Hawkeye/
├── backend/                    # 🖥️ Flask API Server
│   ├── agents/                 # AI agent system
│   ├── models/                 # Model definitions
│   ├── routes/                 # API endpoints
│   └── services/               # Business logic
│
├── frontend/                   # 🌐 Next.js Web App
│   └── src/
│
├── data/                       # 📊 Data (Git ignored)
│   ├── raw/                    # Original datasets
│   │   ├── PD4T/              # PD4T dataset
│   │   └── TULIP/             # TULIP dataset
│   ├── processed/              # Preprocessed data
│   │   ├── features/          # Extracted features
│   │   └── cache/             # Cached computations
│   └── external/               # External reference data
│
├── models/                     # 💾 Trained Models (Git ignored)
│   └── trained/                # Production models
│
├── scripts/                    # 🔧 Utility Scripts
│   ├── training/               # Model training scripts
│   ├── evaluation/             # Evaluation scripts
│   └── hpc/                    # HPC cluster scripts
│
├── experiments/                # 🧪 Experiment Management
│   ├── configs/                # Experiment configurations
│   └── results/                # Experiment outputs
│
├── notebooks/                  # 📓 Jupyter Notebooks
├── docs/                       # 📚 Documentation
└── frontend/public/data/       # De-identified skeleton/keypoint demo data
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA (optional, for GPU acceleration)

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/hawkeye.git
cd hawkeye

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### 2. Download Datasets

Download datasets separately (not included in repo):

```bash
# PD4T Dataset
# Place in: data/raw/PD4T/

# TULIP Dataset
# Place in: data/raw/TULIP/
```

### 3. Run Development

```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

Access at: http://localhost:3000

## 🔬 Training Models

```bash
# Train LSTM model for gait analysis
python scripts/training/train_gait_lstm.py

# Train finger tapping model
python scripts/training/train_finger_tapping_ml.py

# Evaluate on test set
python scripts/evaluation/compare_scores_video.py
```

## 📊 Datasets

### PD4T (Parkinson's Disease 4 Tasks)
- **2,931 videos** from 30 patients
- 4 motor tasks from MDS-UPDRS
- Expert-rated severity scores (0-4)
- See: `docs/PD4T_Analysis_Report.md`

### TULIP
- Additional PD assessment dataset
- Complementary motor tasks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                    │
├─────────────────────────────────────────────────────────┤
│                    Backend API (Flask)                   │
├────────────┬────────────┬────────────┬─────────────────┤
│  Vision    │  Clinical  │   Report   │   Orchestrator  │
│   Agent    │   Agent    │   Agent    │                 │
├────────────┴────────────┴────────────┴─────────────────┤
│              ML Models (LSTM, XGBoost, RF)              │
├─────────────────────────────────────────────────────────┤
│              MediaPipe Pose Estimation                   │
└─────────────────────────────────────────────────────────┘
```

## 📈 Performance

| Model | Task | Accuracy | MAE |
|-------|------|----------|-----|
| LSTM | Gait | 72% | 0.35 |
| RF | Finger Tapping | 68% | 0.42 |
| XGBoost | Hand Movement | 65% | 0.48 |

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests: `pytest`
4. Submit PR

## 📄 License

[License TBD]

## 📞 Contact

- **Team**: Hawkeye Research Team
- **Email**: [contact@example.com]
