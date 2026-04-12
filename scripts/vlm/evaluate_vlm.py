"""
VLM Evaluation Script for PD4T Dataset
Supports: Qwen2-VL, LLaVA, GPT-4V (API)

Usage:
    # Local (API-based)
    python scripts/vlm/evaluate_vlm.py --config experiments/configs/vlm/qwen_vl_evaluation.yaml --api

    # HPC (Local model)
    python scripts/vlm/evaluate_vlm.py --config experiments/configs/vlm/qwen_vl_evaluation.yaml
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_config import (
    CURRENT_ENV, Environment, PD4T_ROOT, RESULTS_DIR,
    VLM_MODELS_DIR, GPU_CONFIG, print_config
)

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLMPrediction:
    """VLM prediction result"""
    video_id: str
    task: str
    gt_score: int
    pred_score: int
    confidence: float
    explanation: str
    raw_response: str


class VLMEvaluator:
    """Base class for VLM evaluation"""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.processor = None

    def load_model(self):
        raise NotImplementedError

    def predict(self, video_path: str, task: str) -> VLMPrediction:
        raise NotImplementedError

    def get_prompt(self, task: str) -> str:
        """Generate task-specific prompt"""
        prompts = self.config.get("prompts", {})
        system = prompts.get("system", "").format(task=task)
        instruction = prompts.get("scoring_instruction", "").format(task=task)
        task_specific = prompts.get("task_specific", {}).get(task, "")

        return f"{system}\n\n{instruction}\n\n{task_specific}"


class LocalVLMEvaluator(VLMEvaluator):
    """Local VLM model evaluator (for HPC)"""

    def load_model(self):
        """Load local VLM model with quantization"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        model_name = self.config["model"]["name"]
        quantization = self.config["model"].get("quantization", None)

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Quantization: {quantization}")

        load_kwargs = {
            "device_map": self.config["model"].get("device_map", "auto"),
            "trust_remote_code": True,
        }

        # Apply quantization
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            load_kwargs["load_in_8bit"] = True

        # Set dtype
        dtype = self.config["model"].get("torch_dtype", "bfloat16")
        if dtype == "bfloat16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype == "float16":
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        logger.info("Model loaded successfully")

    def predict(self, video_path: str, task: str, gt_score: int) -> VLMPrediction:
        """Generate prediction for a video"""
        import torch

        # Sample frames from video
        frames = self._sample_frames(video_path)

        # Generate prompt
        prompt = self.get_prompt(task)

        # Process input
        inputs = self.processor(
            text=prompt,
            images=frames,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["model"].get("max_new_tokens", 256),
                do_sample=False,
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Parse score from response
        pred_score, confidence = self._parse_score(response)

        return VLMPrediction(
            video_id=Path(video_path).stem,
            task=task,
            gt_score=gt_score,
            pred_score=pred_score,
            confidence=confidence,
            explanation=response,
            raw_response=response,
        )

    def _sample_frames(self, video_path: str) -> List:
        """Sample frames from video"""
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = self.config["data"]["video_sampling"]["num_frames"]

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def _parse_score(self, response: str) -> Tuple[int, float]:
        """Parse UPDRS score from VLM response"""
        import re

        # Look for patterns like "Score: 2" or "UPDRS score is 2"
        patterns = [
            r"score[:\s]+(\d)",
            r"rating[:\s]+(\d)",
            r"(\d)\s*(?:out of|/)\s*4",
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 4), 0.8  # Clamp to 0-4

        # Default: couldn't parse
        return 2, 0.3  # Return middle score with low confidence


class APIVLMEvaluator(VLMEvaluator):
    """API-based VLM evaluator (GPT-4V, Claude)"""

    def load_model(self):
        """Initialize API client"""
        api_model = self.config["environment"]["local"]["api_model"]

        if "gpt" in api_model:
            from openai import OpenAI
            self.client = OpenAI()
            self.api_type = "openai"
        elif "claude" in api_model:
            from anthropic import Anthropic
            self.client = Anthropic()
            self.api_type = "anthropic"
        else:
            raise ValueError(f"Unknown API model: {api_model}")

        logger.info(f"Using API: {api_model}")

    def predict(self, video_path: str, task: str, gt_score: int) -> VLMPrediction:
        """Generate prediction using API"""
        import base64

        # Sample frames and encode
        frames = self._sample_frames(video_path)
        encoded_frames = [self._encode_image(f) for f in frames]

        prompt = self.get_prompt(task)

        if self.api_type == "openai":
            response = self._call_openai(prompt, encoded_frames)
        else:
            response = self._call_anthropic(prompt, encoded_frames)

        pred_score, confidence = self._parse_score(response)

        return VLMPrediction(
            video_id=Path(video_path).stem,
            task=task,
            gt_score=gt_score,
            pred_score=pred_score,
            confidence=confidence,
            explanation=response,
            raw_response=response,
        )

    def _encode_image(self, image) -> str:
        """Encode PIL image to base64"""
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _call_openai(self, prompt: str, images: List[str]) -> str:
        """Call OpenAI GPT-4V API"""
        content = [{"type": "text", "text": prompt}]
        for img in images[:4]:  # GPT-4V limited images
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
        )
        return response.choices[0].message.content

    # ... (anthropic implementation similar)

    def _sample_frames(self, video_path: str) -> List:
        """Sample frames from video"""
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = min(self.config["data"]["video_sampling"]["num_frames"], 8)

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def _parse_score(self, response: str) -> Tuple[int, float]:
        """Parse UPDRS score from VLM response"""
        import re

        patterns = [
            r"score[:\s]+(\d)",
            r"rating[:\s]+(\d)",
            r"(\d)\s*(?:out of|/)\s*4",
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 4), 0.8

        return 2, 0.3


def evaluate(config_path: str, use_api: bool = False):
    """Run VLM evaluation"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print_config()

    # Select evaluator
    if use_api or CURRENT_ENV == Environment.LOCAL:
        evaluator = APIVLMEvaluator(config)
    else:
        evaluator = LocalVLMEvaluator(config)

    evaluator.load_model()

    # Load test data
    results = []
    for task in config["data"]["tasks"]:
        logger.info(f"Evaluating task: {task}")

        # Load annotations
        task_dir = task.replace("_", " ").title() if "_" in task else task.capitalize()
        annotation_path = PD4T_ROOT / "Annotations" / task_dir / "test.csv"

        if not annotation_path.exists():
            logger.warning(f"Annotation file not found: {annotation_path}")
            continue

        df = pd.read_csv(annotation_path, header=None, names=["video_id", "frames", "score"])

        for _, row in tqdm(df.iterrows(), total=len(df), desc=task):
            video_id = row["video_id"]
            gt_score = row["score"]

            # Find video file
            video_dir = PD4T_ROOT / "Videos" / task_dir
            video_path = find_video(video_dir, video_id)

            if video_path is None:
                logger.warning(f"Video not found: {video_id}")
                continue

            try:
                pred = evaluator.predict(str(video_path), task, gt_score)
                results.append(pred)
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")

    # Calculate metrics
    if results:
        calculate_and_save_metrics(results, config)


def find_video(video_dir: Path, video_id: str) -> Optional[Path]:
    """Find video file by ID"""
    for folder in video_dir.iterdir():
        if folder.is_dir():
            for ext in ['.mp4', '.avi', '.mov']:
                video_path = folder / f"{video_id}{ext}"
                if video_path.exists():
                    return video_path
    return None


def calculate_and_save_metrics(results: List[VLMPrediction], config: Dict):
    """Calculate and save evaluation metrics"""
    from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score

    gt_scores = [r.gt_score for r in results]
    pred_scores = [r.pred_score for r in results]

    metrics = {
        "accuracy": accuracy_score(gt_scores, pred_scores),
        "mae": mean_absolute_error(gt_scores, pred_scores),
        "within_one": np.mean(np.abs(np.array(gt_scores) - np.array(pred_scores)) <= 1),
        "kappa": cohen_kappa_score(gt_scores, pred_scores, weights="quadratic"),
        "total_samples": len(results),
    }

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Save results
    output_dir = RESULTS_DIR / "vlm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / config["output"]["metrics_file"], 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame([
        {
            "video_id": r.video_id,
            "task": r.task,
            "gt_score": r.gt_score,
            "pred_score": r.pred_score,
            "confidence": r.confidence,
        }
        for r in results
    ])
    pred_df.to_csv(output_dir / config["output"]["prediction_file"], index=False)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Evaluation for PD4T")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--api", action="store_true", help="Use API instead of local model")

    args = parser.parse_args()
    evaluate(args.config, args.api)
