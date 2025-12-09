# Gait Task Evaluation Report: GPT-4o vs Gemini 2.0 Flash

**Evaluation Date**: 2025-11-25
**Dataset**: PD4T Test Set (User-labeled, 73 Gait samples)
**Models Compared**:
- GPT-4o (OpenAI)
- Gemini 2.0 Flash Experimental v3 (Google, medical terms removed)

---

## Executive Summary

**Winner: GPT-4o** with 64.4% accuracy vs Gemini's 54.8% (+9.6 percentage points)

### Key Findings
- ✅ **GPT-4o excels at detecting normal gait (GT=0)**: 78.8% accuracy
- ⚠️ **Gemini shows severe score 1 bias**: 86.3% of predictions are score 1
- ✅ **Gemini performs better on GT=1 and GT=2**: 96.9% and 37.5% respectively
- 📊 **GPT-4o has balanced prediction distribution**, Gemini does not

---

## 1. Overall Performance Metrics

| Metric | GPT-4o | Gemini 2.0 Flash v3 | Winner | Difference |
|--------|--------|---------------------|--------|------------|
| **Accuracy** | **64.4%** (47/73) | 54.8% (40/73) | **GPT-4o** | +9.6%p |
| **MAE (Mean Absolute Error)** | **0.397** | 0.452 | **GPT-4o** | -0.055 |
| **Valid Predictions** | 73/73 (100%) | 73/73 (100%) | Tie | - |
| **Failed Predictions** | 0 | 0 | Tie | - |

---

## 2. Prediction Distribution Analysis

### 2.1 GPT-4o Distribution (Balanced)

| Predicted Score | Count | Percentage | Assessment |
|----------------|-------|------------|------------|
| 0 (Normal) | 40 | 54.8% | ✅ Balanced |
| 1 (Slight) | 31 | 42.5% | ✅ Balanced |
| 2 (Mild) | 2 | 2.7% | ⚠️ Underused |
| 3 (Moderate) | 0 | 0% | ❌ Never used |
| 4 (Severe) | 0 | 0% | ❌ Never used |

**Analysis**: GPT-4o shows good balance between scores 0 and 1, reflecting the ground truth distribution.

### 2.2 Gemini 2.0 Flash v3 Distribution (Biased)

| Predicted Score | Count | Percentage | Assessment |
|----------------|-------|------------|------------|
| 0 (Normal) | 6 | 8.2% | ❌ Severely underused |
| **1 (Slight)** | **63** | **86.3%** | ⚠️ **Extreme bias** |
| 2 (Mild) | 4 | 5.5% | ⚠️ Underused |
| 3 (Moderate) | 0 | 0% | ❌ Never used |
| 4 (Severe) | 0 | 0% | ❌ Never used |

**Critical Issue**: Gemini defaults to score 1 in 86.3% of cases, indicating a "safe middle value" bias.

---

## 3. Ground Truth Distribution

| GT Score | Count | Percentage | Clinical Interpretation |
|----------|-------|------------|------------------------|
| 0 (Normal) | 33 | 45.2% | Healthy gait |
| 1 (Slight) | 32 | 43.8% | Minor motor impairment |
| 2 (Mild) | 8 | 11.0% | Moderate motor impairment |
| 3+ | 0 | 0% | No severe cases in test set |

---

## 4. Confusion Matrix Comparison

### 4.1 GPT-4o Confusion Matrix

|  | Predicted 0 | Predicted 1 | Predicted 2 | Total | Accuracy |
|--|-------------|-------------|-------------|-------|----------|
| **GT 0** | **26** ✅ | 7 | 0 | 33 | **78.8%** |
| **GT 1** | 11 | **20** ✅ | 1 | 32 | **62.5%** |
| **GT 2** | 3 | 4 | **1** ✅ | 8 | **12.5%** |

**Key Observations**:
- Strong performance on GT=0 (26/33 correct)
- Tendency to predict score 0 for GT=1 cases (11 false negatives)
- Poor performance on GT=2 (only 1/8 correct)

### 4.2 Gemini 2.0 Flash v3 Confusion Matrix

|  | Predicted 0 | Predicted 1 | Predicted 2 | Total | Accuracy |
|--|-------------|-------------|-------------|-------|----------|
| **GT 0** | **6** ✅ | 27 ❌ | 0 | 33 | **18.2%** |
| **GT 1** | 0 | **31** ✅ | 1 | 32 | **96.9%** |
| **GT 2** | 0 | 5 ❌ | **3** ✅ | 8 | **37.5%** |

**Critical Issues**:
- Severe misclassification of GT=0 cases (27/33 predicted as 1)
- Excellent performance on GT=1 (31/32 correct)
- Improved GT=2 performance vs GPT-4o (3/8 vs 1/8)

---

## 5. Per-Class Performance Analysis

### Class 0 (Normal Gait)

| Model | Precision | Recall | F1-Score | Samples |
|-------|-----------|--------|----------|---------|
| **GPT-4o** | 65.0% (26/40) | **78.8%** (26/33) | 71.2% | 33 |
| **Gemini** | 100% (6/6) | 18.2% (6/33) | 30.8% | 33 |

**Winner: GPT-4o** - Much better at detecting normal gait (+60.6%p recall)

### Class 1 (Slight Impairment)

| Model | Precision | Recall | F1-Score | Samples |
|-------|-----------|--------|----------|---------|
| **GPT-4o** | 64.5% (20/31) | 62.5% (20/32) | 63.5% | 32 |
| **Gemini** | 49.2% (31/63) | **96.9%** (31/32) | 65.2% | 32 |

**Winner: Gemini** - Excellent GT=1 detection but with many false positives

### Class 2 (Mild Impairment)

| Model | Precision | Recall | F1-Score | Samples |
|-------|-----------|--------|----------|---------|
| **GPT-4o** | 50.0% (1/2) | 12.5% (1/8) | 20.0% | 8 |
| **Gemini** | 75.0% (3/4) | **37.5%** (3/8) | 50.0% | 8 |

**Winner: Gemini** - Better at detecting moderate impairment (+25%p recall)

---

## 6. Error Analysis

### 6.1 GPT-4o Error Patterns

**Most Common Errors**:
1. **GT=1 → Pred=0 (11 cases)**: Underestimating slight impairment as normal
2. **GT=2 → Pred=1 (4 cases)**: Underestimating mild impairment as slight
3. **GT=0 → Pred=1 (7 cases)**: Overestimating normal gait

**Bias**: Slight tendency to predict score 0, possibly too optimistic on normal cases

### 6.2 Gemini 2.0 Flash v3 Error Patterns

**Most Common Errors**:
1. **GT=0 → Pred=1 (27 cases)**: Severe overestimation of normal gait ⚠️
2. **GT=2 → Pred=1 (5 cases)**: Underestimating mild impairment

**Critical Bias**: Model defaults to score 1 as "safe middle value", avoiding extreme predictions

---

## 7. Clinical Implications

### 7.1 GPT-4o Clinical Performance

**Strengths**:
- ✅ Good sensitivity for normal gait detection (78.8%)
- ✅ Balanced predictions reduce systematic bias
- ✅ Lower false positive rate for impairment

**Weaknesses**:
- ⚠️ May miss some slight impairments (11 GT=1 → Pred=0)
- ❌ Poor detection of moderate impairment (12.5% recall)

**Clinical Risk**: May underestimate disease progression in borderline cases

### 7.2 Gemini Clinical Performance

**Strengths**:
- ✅ Excellent sensitivity for slight impairment (96.9%)
- ✅ Better detection of moderate impairment vs GPT-4o

**Weaknesses**:
- ❌ **Critical: 82% false positive rate on normal gait** (27/33)
- ⚠️ Conservative bias leads to over-diagnosis
- ❌ Lacks confidence in extreme predictions

**Clinical Risk**: High false positive rate may lead to unnecessary interventions and patient anxiety

---

## 8. Statistical Significance

### Cohen's Kappa (Inter-rater Agreement)

| Model | Kappa Score | Agreement Level |
|-------|-------------|-----------------|
| GPT-4o | 0.424 | Moderate agreement |
| Gemini 2.0 Flash v3 | 0.264 | Fair agreement |

**Winner: GPT-4o** - Better agreement with ground truth annotations

---

## 9. Prompt Comparison

### GPT-4o Prompt (Original)
```
You are an expert neurologist specializing in Parkinson's Disease.
Analyze the video of a patient performing the 'Gait' task.
Assess the stride amplitude, stride speed, height of foot lift, heel strike, turning, and arm swing.

Rate the severity of the motor impairment on the MDS-UPDRS scale:
0: Normal (No problems)
1: Slight (Slight slowness/small amplitude, no decrement)
2: Mild (Mild slowness/amplitude, some decrement or hesitations)
3: Moderate (Moderate slowness/amplitude, frequent hesitations/halts)
4: Severe (Severe impairment, barely performs the task)
```

### Gemini 2.0 Flash v3 Prompt (Modified - Medical Terms Removed)
```
You are an expert in movement analysis.
Analyze the video of a person performing the 'Gait' task.
Assess the stride amplitude, stride speed, height of foot lift, heel strike, turning, and arm swing.

Rate the severity of the motor performance on a 0-4 scale:
0: Normal (No problems)
1: Slight (Slight slowness/small amplitude, no decrement)
2: Mild (Mild slowness/amplitude, some decrement or hesitations)
3: Moderate (Moderate slowness/amplitude, frequent hesitations/halts)
4: Severe (Severe impairment, barely performs the task)
```

**Impact**: Removing "neurologist" and "Parkinson's Disease" did not eliminate Gemini's score 1 bias

---

## 10. Technical Implementation Differences

| Aspect | GPT-4o | Gemini 2.0 Flash v3 |
|--------|--------|---------------------|
| **Input Method** | Frame sampling (16 frames) | Full video upload |
| **Context Window** | Limited to sampled frames | Complete temporal context |
| **Processing Time** | ~17s per video | ~17s per video |
| **API Cost** | ~$0.01 per video (paid) | Free tier available |
| **Failure Rate** | 0% | 0% |
| **Safety Filters** | Not triggered | Not triggered (after prompt modification) |

---

## 11. Recommendations

### For Clinical Use

1. **GPT-4o is recommended for screening** due to:
   - Better overall accuracy (64.4% vs 54.8%)
   - Lower false positive rate on normal cases
   - More balanced predictions

2. **Gemini could be used as second opinion** for:
   - Detecting subtle impairments (GT=1)
   - Reducing false negatives in slight impairment cases

3. **Ensemble approach** combining both models may improve robustness

### For Model Improvement

#### GPT-4o Improvements
- Improve detection of moderate impairment (GT=2)
- Reduce false negatives for GT=1 cases
- Consider using full video instead of frame sampling

#### Gemini Improvements
- **Critical**: Address score 1 bias through prompt engineering or fine-tuning
- Improve confidence calibration for extreme scores
- Add few-shot examples showing diverse score distributions

---

## 12. Conclusions

### Overall Winner: GPT-4o

**Reasons**:
1. ✅ **9.6 percentage point accuracy advantage** (64.4% vs 54.8%)
2. ✅ **Balanced prediction distribution** reflecting ground truth
3. ✅ **Superior normal gait detection** (78.8% vs 18.2%)
4. ✅ **Lower false positive rate** for clinical deployment

### Gemini's Critical Flaw

The **86.3% score 1 bias** is a fundamental issue that outweighs its advantages in GT=1 and GT=2 detection. This conservative prediction pattern suggests the model lacks confidence in differentiating severity levels.

### Next Steps

1. Complete evaluation on remaining tasks (Finger tapping, Hand movement, Leg agility)
2. Perform full 485-sample comparison across all tasks
3. Investigate prompt engineering to reduce Gemini's score 1 bias
4. Test ensemble approaches combining both models

---

## Appendix: Raw Data Summary

**Test Set**: 73 Gait videos
**Ground Truth**: 33 normal (GT=0), 32 slight (GT=1), 8 mild (GT=2)
**GPT-4o**: 47/73 correct (64.4%), 0 failures
**Gemini 2.0 Flash v3**: 40/73 correct (54.8%), 0 failures

**Report Generated**: 2025-11-25
**Analysis Tool**: Python pandas, scikit-learn
**Code Repository**: VLM_commercial/scripts/
