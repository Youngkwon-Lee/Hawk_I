# Hawkeye Development Roadmap
**Current Best: Mamba + Enhanced Features (Pearson 0.609)**

---

## 🎯 Overall Strategy

### Goal: Pearson 0.70+ (Clinical-grade accuracy)

| Approach | Expected Gain | Difficulty | Time | Priority |
|----------|--------------|------------|------|----------|
| **Advanced Features** | +10-15% | Low | 1-2 weeks | ⭐⭐⭐ HIGH |
| **Model Architecture** | +5-10% | Medium | 1-2 weeks | ⭐⭐ MEDIUM |
| **Ensemble** | +3-5% | Low | 3-5 days | ⭐⭐ MEDIUM |
| **Multi-task Learning** | +8-12% | High | 2-3 weeks | ⭐ LOW |
| **VideoMamba (RGB)** | +15-20% | High | 3-4 weeks | ⭐⭐⭐ HIGH |

---

## 1. Feature Engineering (HIGHEST PRIORITY) 🔥

### A. Quick Wins (1-2 days) → Pearson 0.64
**Implementation:**
```python
# 4가지 핵심 features
1. amplitude_decline_rate  # Fatigue indicator
2. frequency_variability   # Rhythm consistency
3. smoothness_sparc        # Movement quality
4. hesitation_count        # Freezing detection
```

**Script:** `scripts/training/train_finger_clinical_features.py`

**Expected Results:**
- Pearson: 0.609 → **0.64** (+5%)
- MAE: 0.444 → **0.43** (-3%)

---

### B. Clinical Features (3-5 days) → Pearson 0.68
**Implementation:**
```python
# UPDRS-aligned features
- amplitude_score (0-4)
- speed_score
- rhythm_score
- halts_score
- fatigue_index

# Joint angles (MediaPipe 3D)
- thumb_index_angle
- wrist_flexion_angle
- angular_velocity
```

**Script:** `scripts/training/train_finger_clinical_v2.py`

**Expected Results:**
- Pearson: 0.64 → **0.68** (+6%)
- MAE: 0.43 → **0.41** (-5%)
- **Clinical interpretability ↑**

---

## 2. Model Architecture Improvements

### A. Full Mamba Implementation
**Current Issue:**
- Simplified selective scan (parallel approximation)
- Sequence length truncation (100 frames)

**Solution:**
```python
# Install official Mamba
pip install mamba-ssm

# Use full selective scan
from mamba_ssm import Mamba

class FullMamba(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
```

**Expected Gain:** Pearson +2-3% → **0.63**

---

### B. Mamba-2 (2024)
**Improvements:**
- State Space Duality (SSD)
- 8x faster training
- Better long sequences

**Reference:** [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)

**Expected Gain:** Pearson +3-5% → **0.64**

---

### C. Bidirectional Mamba
```python
class BiMamba(nn.Module):
    def __init__(self, d_model):
        self.forward_mamba = Mamba(d_model)
        self.backward_mamba = Mamba(d_model)

    def forward(self, x):
        x_fwd = self.forward_mamba(x)
        x_bwd = self.backward_mamba(x.flip(dims=[1]))
        return (x_fwd + x_bwd.flip(dims=[1])) / 2
```

**Expected Gain:** Pearson +2-4% → **0.63**

---

## 3. Ensemble Strategies

### A. Mamba + ST-GCN Ensemble
**Rationale:**
- Mamba: Temporal patterns (best)
- ST-GCN: Spatial patterns (2nd best)
- Complementary strengths

**Implementation:**
```python
class MambaSTGCNEnsemble(nn.Module):
    def __init__(self):
        self.mamba = Mamba(...)
        self.stgcn = STGCN(...)
        self.fusion = nn.Linear(2, 1)

    def forward(self, x):
        mamba_pred = self.mamba(x)
        stgcn_pred = self.stgcn(x)
        return self.fusion(torch.stack([mamba_pred, stgcn_pred], dim=-1))
```

**Expected Gain:** Pearson +3-5% → **0.64**

**Time:** 2-3 days

---

### B. Multi-scale Ensemble
```python
# Different temporal resolutions
- Mamba (full sequence)
- Mamba (1/2 downsampled)
- Mamba (1/4 downsampled)

# Vote or weighted average
```

**Expected Gain:** Pearson +2-3%

---

## 4. Multi-task Learning

### A. Joint Training (Finger + Gait)
**Architecture:**
```python
class MultiTaskMamba(nn.Module):
    def __init__(self):
        # Shared encoder
        self.shared_encoder = Mamba(...)

        # Task-specific heads
        self.finger_head = nn.Linear(256, 1)
        self.gait_head = nn.Linear(256, 1)

    def forward(self, x, task):
        h = self.shared_encoder(x)
        if task == 'finger':
            return self.finger_head(h)
        else:
            return self.gait_head(h)
```

**Benefits:**
- Cross-task knowledge transfer
- Better generalization
- Single model for both tasks

**Expected Gain:** Pearson +5-8% → **0.65**

**Time:** 1-2 weeks

---

### B. Auxiliary Tasks
```python
# Main task: UPDRS score prediction
# Auxiliary: Movement phase detection, severity classification

class AuxiliaryMamba(nn.Module):
    def forward(self, x):
        h = self.encoder(x)

        # Main
        score_pred = self.score_head(h)

        # Auxiliary
        phase_pred = self.phase_head(h)  # Open/Close/Hold
        severity_pred = self.severity_head(h)  # Mild/Moderate/Severe

        return score_pred, phase_pred, severity_pred
```

**Expected Gain:** Pearson +3-5%

---

## 5. VideoMamba (RGB Video) 🚀

### A. Direct RGB Processing
**Current:** Skeleton only (MediaPipe → Mamba)
**Proposed:** RGB video → VideoMamba

**Advantages:**
- **Tremor detection**: 미세한 떨림 포착
- **Skin texture**: 근육 긴장, 피부 색 변화
- **Context**: 전체적인 움직임 패턴

**Architecture:**
```python
# VideoMamba for PD assessment
class VideoMambaForPD(nn.Module):
    def __init__(self):
        # Patch embedding (video frames)
        self.patch_embed = PatchEmbed3D(...)

        # VideoMamba encoder
        self.video_mamba = VideoMamba(...)

        # Classifier
        self.head = nn.Linear(768, 1)
```

**Reference:** [VideoMamba Paper](https://arxiv.org/abs/2403.06977)

**Expected Gain:** Pearson +15-20% → **0.72+**

**Challenges:**
- GPU memory (V100 16GB may be insufficient)
- Training time (10x longer)
- Data size (10x larger)

**Solutions:**
- Use A100 40GB or H100
- Gradient checkpointing
- Mixed precision (FP16)

**Time:** 3-4 weeks

---

## 6. Data Augmentation 2.0

### A. Advanced Time Series Augmentation
**Current:** Time warping, jittering, scaling

**Add:**
```python
# Mixup for time series
def mixup_timeseries(x1, x2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2

# CutMix for sequences
def cutmix_sequences(x1, x2, beta=1.0):
    lam = np.random.beta(beta, beta)
    cut_len = int(len(x1) * lam)
    start = np.random.randint(0, len(x1) - cut_len)

    x_mixed = x1.copy()
    x_mixed[start:start+cut_len] = x2[start:start+cut_len]
    return x_mixed

# Physics-informed augmentation
def apply_tremor(x, freq=5, amp=0.01):
    """Add realistic parkinsonian tremor"""
    t = np.arange(len(x)) / 30.0  # 30 fps
    tremor = amp * np.sin(2 * np.pi * freq * t)
    return x + tremor[:, None]
```

**Expected Gain:** Pearson +2-4%

---

### B. GAN-based Augmentation
```python
# Generate synthetic PD movement patterns
class MovementGAN(nn.Module):
    def __init__(self):
        self.generator = Generator(...)
        self.discriminator = Discriminator(...)

    # Train on real PD data
    # Generate synthetic samples for minority classes (score 0, 4)
```

**Expected Gain:** Pearson +3-5%
**Time:** 2-3 weeks

---

## 7. Uncertainty Quantification

### A. Bayesian Mamba
```python
# Monte Carlo Dropout
class BayesianMamba(nn.Module):
    def predict_with_uncertainty(self, x, n_samples=100):
        preds = []
        for _ in range(n_samples):
            pred = self.forward(x)  # Dropout active
            preds.append(pred)

        mean = np.mean(preds)
        std = np.std(preds)  # Uncertainty

        return mean, std
```

**Benefits:**
- Confidence scores for predictions
- Identify ambiguous cases
- Clinical decision support

---

### B. Ensemble Uncertainty
```python
# Disagreement between models
predictions = [model1(x), model2(x), model3(x)]
uncertainty = np.std(predictions)
```

---

## 8. Explainability (XAI)

### A. Attention Visualization
```python
# Which frames are most important?
class MambaWithAttention(nn.Module):
    def forward(self, x):
        h, attention_weights = self.mamba(x, return_attention=True)
        return h, attention_weights

# Visualize
plt.plot(attention_weights)
plt.xlabel('Frame')
plt.ylabel('Importance')
```

---

### B. SHAP for Feature Importance
```python
import shap

explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# Top features
shap.summary_plot(shap_values, X_test)
```

**Clinical Value:** 의사가 결과를 신뢰할 수 있음

---

## 9. Real-world Deployment

### A. Model Optimization
```python
# ONNX export
torch.onnx.export(model, dummy_input, "mamba.onnx")

# TensorRT optimization
import tensorrt as trt
```

**Target:** 30 FPS real-time inference

---

### B. Edge Deployment
```python
# Quantization (FP32 → INT8)
from torch.quantization import quantize_dynamic

model_int8 = quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Size: 100MB → 25MB
# Speed: 2x faster
```

**Target:** 스마트폰에서 실행 가능

---

## 10. Dataset Expansion

### A. External Datasets
- **mPower** (Sage Bionetworks): 10,000+ PD patients
- **PPMI** (Parkinson's Progression Markers Initiative)
- **Real-world data**: Clinic recordings

**Expected Gain:** Pearson +5-10% (더 많은 데이터)

---

### B. Longitudinal Data
```python
# Track patient over time
# Predict disease progression
class ProgressionPredictor(nn.Module):
    def predict_future_score(self, history):
        # Input: [t0, t1, t2, t3] scores
        # Output: t4 predicted score
```

---

## Summary: Roadmap Priorities

### 🔥 High Priority (Next 2 weeks)
1. **Advanced Features** → Pearson 0.68 (+9%)
2. **Mamba + ST-GCN Ensemble** → +3-5%
3. **Full Mamba implementation** → +2-3%

**Total Expected:** Pearson **0.70+**

---

### 🚀 Medium Priority (1-2 months)
1. **VideoMamba** → +15-20%
2. **Multi-task Learning** → +5-8%
3. **Dataset Expansion** → +5-10%

**Total Expected:** Pearson **0.75+**

---

### 💡 Long-term (3-6 months)
1. **Real-world Deployment**
2. **Longitudinal prediction**
3. **Clinical validation study**

**Goal:** FDA approval, clinical adoption

---

## Implementation Plan (Next Sprint)

### Week 1: Clinical Features
- [ ] Implement amplitude_decline_rate
- [ ] Add frequency_variability
- [ ] Calculate SPARC smoothness
- [ ] Detect hesitations
- [ ] Train Mamba + Clinical Features
- [ ] **Target: Pearson 0.64**

### Week 2: Advanced Features
- [ ] Add UPDRS-aligned scores
- [ ] Extract joint angles (3D)
- [ ] Calculate fatigue index
- [ ] Train Mamba + Full Clinical
- [ ] **Target: Pearson 0.68**

### Week 3: Ensemble
- [ ] Implement Mamba + ST-GCN ensemble
- [ ] Test multi-scale ensemble
- [ ] **Target: Pearson 0.70**

### Week 4: Validation
- [ ] Cross-validation on Gait task
- [ ] External validation (if dataset available)
- [ ] Performance analysis
- [ ] Write paper draft

---

## Expected Timeline to Pearson 0.70

| Week | Milestone | Pearson | Cumulative Gain |
|------|-----------|---------|-----------------|
| 0 | Current (Mamba + Enhanced) | 0.609 | - |
| 1 | + Clinical Features (Quick) | 0.64 | +5% |
| 2 | + Advanced Clinical | 0.68 | +11% |
| 3 | + Ensemble | 0.70 | +15% |
| 4 | Validation & Refinement | **0.70+** | **+15%** |

---

## Success Metrics

### Technical
- ✅ Pearson > 0.70
- ✅ MAE < 0.40
- ✅ Exact accuracy > 65%

### Clinical
- ✅ Agreement with expert raters (ICC > 0.80)
- ✅ Sensitivity/Specificity > 85%
- ✅ Explainable predictions

### Deployment
- ✅ Real-time inference (< 100ms)
- ✅ Mobile-ready (< 50MB model)
- ✅ API integration complete

---

## References

1. **Mamba-2**: Dao & Gu (2024) - "Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality"
2. **VideoMamba**: Li et al. (2024) - "VideoMamba: State Space Model for Efficient Video Understanding"
3. **Clinical Features**: Espay et al. (2011) - "Differential response of speed, amplitude, and rhythm to dopaminergic medications"
4. **SPARC**: Balasubramanian et al. (2015) - "On the analysis of movement smoothness"
5. **Multi-task Learning**: Caruana (1997) - "Multitask Learning"
