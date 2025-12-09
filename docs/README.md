# VLM Commercial Models Evaluation for PD4T

This directory contains evaluation scripts for commercial VLM (Vision-Language Models) on the PD4T Parkinson's Disease dataset.

## 📁 Folder Structure

```
VLM_commercial/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── configs/                           # Configuration files
│   └── api_keys.env.example          # API keys template
├── scripts/                           # Evaluation scripts
│   ├── gpt4o_vlm_evaluation.py       # GPT-4o evaluation
│   ├── gemini_2_flash_evaluation.py  # Gemini 2.0 Flash evaluation
│   └── compare_results.py            # Compare and analyze results
├── models/                            # Model-specific configs
│   ├── gpt4o_config.json
│   └── gemini_config.json
├── results/                           # Evaluation results
│   ├── gpt4o_results.csv
│   ├── gemini_2_flash_results.csv
│   └── comparison_report.md
├── logs/                              # Execution logs
└── data_mapping/                      # Test data mappings
    ├── fingertapping_test.csv
    ├── gait_test.csv
    ├── handmovement_test.csv
    └── legagility_test.csv
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Keys

```bash
# Copy template and add your keys
cp configs/api_keys.env.example configs/api_keys.env

# Edit configs/api_keys.env
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
```

### 3. Run Evaluations

#### GPT-4o Evaluation
```bash
python scripts/gpt4o_vlm_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --api_key $OPENAI_API_KEY
```

#### Gemini 2.0 Flash Evaluation
```bash
python scripts/gemini_2_flash_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --api_key $GOOGLE_API_KEY
```

#### Evaluate Specific Task
```bash
# GPT-4o on Gait only
python scripts/gpt4o_vlm_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --task "Gait"

# Gemini on Finger tapping only
python scripts/gemini_2_flash_evaluation.py \
    --base_dir ../PD4T/PD4T/PD4T \
    --task "Finger tapping"
```

### 4. Compare Results

```bash
python scripts/compare_results.py \
    --gpt4o_results results/gpt4o_results.csv \
    --gemini_results results/gemini_2_flash_results.csv \
    --output results/comparison_report.md
```

## 📊 Expected Results Format

Each evaluation script generates a CSV with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| task | str | Task name (Gait, Finger tapping, etc.) |
| filename | str | Video identifier |
| gt_score | int | Ground truth MDS-UPDRS score (0-4) |
| pred_score | int | Predicted score (0-4) |
| reason | str | Model's reasoning for the score |
| raw_output | str | Full model output |

## 🔧 Configuration

### GPT-4o Configuration (`models/gpt4o_config.json`)

```json
{
  "model": "gpt-4o",
  "temperature": 0.0,
  "max_tokens": 300,
  "frame_extraction": {
    "Gait": {"max_frames": 16},
    "Finger tapping": {"max_frames": 12},
    "Hand movement": {"max_frames": 12},
    "Leg agility": {"max_frames": 12}
  }
}
```

### Gemini Configuration (`models/gemini_config.json`)

```json
{
  "model": "gemini-2.0-flash-exp",
  "temperature": 0.0,
  "max_output_tokens": 300,
  "safety_settings": {
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"
  }
}
```

## 📈 Evaluation Metrics

The comparison script computes:

- **Accuracy**: Percentage of exact matches
- **MAE (Mean Absolute Error)**: Average score difference
- **Weighted Kappa**: Agreement considering ordinal nature
- **Confusion Matrix**: Per-class performance
- **Class-wise F1-Score**: Handling class imbalance

## 🔍 Test Data Mapping

Test set CSV files are located in `data_mapping/`:

- `gait_test.csv`: 74 samples
- `fingertapping_test.csv`: 136 samples
- `handmovement_test.csv`: 227 samples
- `legagility_test.csv`: 224 samples

Format: `filename,frame_count,gt_score`

Example:
```
15-001760_009,585,1
14-005971_039,613,0
```

## 🌐 API Details

### OpenAI GPT-4o
- **Model**: `gpt-4o` or `gpt-4-vision-preview`
- **Input**: Up to 20 images per request
- **Pricing**: $0.01/image (as of 2025)
- **Rate Limits**: Check your tier

### Google Gemini 2.0 Flash
- **Model**: `gemini-2.0-flash-exp`
- **Input**: Direct video upload
- **Pricing**: Free tier available
- **Rate Limits**: ~60 requests/minute

## 📝 Notes

1. **Cost Estimation**:
   - GPT-4o: ~$30-50 for full PD4T test set (783 videos)
   - Gemini 2.0 Flash: Free (within limits)

2. **Rate Limiting**:
   - Scripts include automatic rate limiting
   - Intermediate results saved every 10 samples

3. **Error Handling**:
   - Videos that fail processing get `-1` score
   - Check logs for detailed error messages

## 🤝 Contributing

For questions or improvements:
- Check logs in `logs/` directory
- Review error handling in scripts
- Compare with reference Qwen implementation

## 📞 Contact

For PD4T dataset questions: a.dadashzadeh@bristol.ac.uk

---

**Last Updated**: 2025-11-24
**PD4T Dataset Version**: Original release
