# VLM Evaluation Command

Vision-Language Model 평가 워크플로우

## 로컬 (API 기반)
```bash
# OpenAI GPT-4V
export OPENAI_API_KEY=your_key
python scripts/vlm/evaluate_vlm.py \
    --config experiments/configs/vlm/qwen_vl_evaluation.yaml \
    --api

# Anthropic Claude 3
export ANTHROPIC_API_KEY=your_key
python scripts/vlm/evaluate_vlm.py \
    --config experiments/configs/vlm/claude_evaluation.yaml \
    --api
```

## HPC (로컬 모델)
```bash
export HAWKEYE_ENV=hpc

# Qwen2-VL-7B (4-bit 양자화)
python scripts/vlm/evaluate_vlm.py \
    --config experiments/configs/vlm/qwen_vl_evaluation.yaml

# LLaVA-1.5-13B
python scripts/vlm/evaluate_vlm.py \
    --config experiments/configs/vlm/llava_evaluation.yaml
```

## VRAM 요구사항
| 모델 | Full | 8-bit | 4-bit |
|------|------|-------|-------|
| Qwen2-VL-7B | 28GB | 14GB | 8GB |
| LLaVA-13B | 52GB | 26GB | 14GB |

## 결과 위치
- Metrics: `experiments/results/vlm/metrics.json`
- Predictions: `experiments/results/vlm/predictions.csv`
