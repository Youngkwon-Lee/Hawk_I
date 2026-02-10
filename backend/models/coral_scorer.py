"""
CORAL Ordinal Scorer - Production Model for UPDRS Assessment

Architecture: CORAL Ordinal Regression with Mamba backbone
Training: 5-Fold Stratified Cross-Validation on PD4T dataset (stratified splits)
Dataset: PD4T with 426 Gait, 806 Finger, 848 Hand, 851 Leg videos

Performance Metrics (Test Set - Pearson Correlation):
- Gait: Pearson 0.790 ✅ Best for movement disorders
- Finger Tapping: Pearson 0.553 ⚠️  Moderate - manual dexterity challenging
- Hand Movement: Pearson 0.598 ✅ Stable - relatively predictable
- Leg Agility: Pearson 0.238 ⚠️  Low - limited data/inherent noise

Note: Experimental models (Mamba+Enhanced) showed higher Pearson in training
(Gait 0.804, Finger 0.609) but CORAL was selected for production due to
consistency across tasks and stability in deployment.

See CLAUDE.md for full experimental vs production comparison.
"""

import os
import math
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. CORAL scorer will not work.")

# Model paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_backend_dir)
MODEL_DIR = os.path.join(_project_root, "models", "trained")


@dataclass
class CORALPrediction:
    """CORAL prediction result"""
    score: int           # UPDRS score 0-4
    confidence: float    # Prediction confidence
    expected_score: float  # Continuous expected value
    probabilities: List[float]  # Class probabilities
    task: str
    model_type: str = "coral_mamba"


# ============================================================
# Mamba Block (must match training architecture exactly)
# ============================================================
if TORCH_AVAILABLE:
    class MambaBlock(nn.Module):
        def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto", d_conv=4):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.expand = expand
            self.d_inner = int(self.expand * self.d_model)

            if dt_rank == "auto":
                self.dt_rank = math.ceil(self.d_model / 16)
            else:
                self.dt_rank = dt_rank

            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                    padding=d_conv-1, groups=self.d_inner)

            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))

            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, x):
            batch, seq_len, dim = x.shape

            xz = self.in_proj(x)
            x_in, z = xz.chunk(2, dim=-1)

            x_conv = x_in.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :seq_len]
            x_in = x_conv.transpose(1, 2)
            x_in = F.silu(x_in)

            x_proj_out = self.x_proj(x_in)
            dt, B, C = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            dt = self.dt_proj(dt)
            dt = F.softplus(dt)

            A = -torch.exp(self.A_log)

            y = self.selective_scan(x_in, dt, A, B, C)
            y = y * F.silu(z)
            output = self.out_proj(y)

            return output

        def selective_scan(self, x, dt, A, B, C):
            batch, seq_len, d_inner = x.shape
            d_state = A.shape[1]

            outputs = []
            state = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

            for t in range(seq_len):
                x_t = x[:, t, :]
                dt_t = dt[:, t, :]
                B_t = B[:, t, :]
                C_t = C[:, t, :]

                dA = torch.exp(torch.clamp(dt_t.unsqueeze(-1) * A.unsqueeze(0), min=-10, max=10))
                dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

                state = state * dA + x_t.unsqueeze(-1) * dB
                state = torch.clamp(state, min=-10, max=10)

                y_t = (state * C_t.unsqueeze(1)).sum(dim=-1)
                y_t = y_t + self.D * x_t

                outputs.append(y_t)

            return torch.stack(outputs, dim=1)


    class MambaCoralModel(nn.Module):
        """Mamba + CORAL Ordinal Regression Model"""

        def __init__(self, input_size, hidden_size=256, num_layers=4,
                     num_classes=5, dropout=0.4):
            super().__init__()

            self.input_proj = nn.Linear(input_size, hidden_size)
            self.layers = nn.ModuleList([
                MambaBlock(hidden_size) for _ in range(num_layers)
            ])
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(num_layers)
            ])
            self.dropout = nn.Dropout(dropout)
            self.coral_head = nn.Linear(hidden_size, num_classes - 1)

        def forward(self, x):
            x = self.input_proj(x)

            for layer, norm in zip(self.layers, self.norms):
                residual = x
                x = norm(x)
                x = layer(x)
                x = self.dropout(x)
                x = x + residual

            x = x.mean(dim=1)
            logits = self.coral_head(x)

            return logits


class CORALScorer:
    """
    CORAL Ordinal Regression Scorer

    Singleton pattern for efficient model loading.
    Supports all 4 MDS-UPDRS tasks.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._configs = {}
            cls._instance._device = None
            cls._instance._loaded = False
        return cls._instance

    def _get_device(self):
        if self._device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self._device = torch.device('cuda')
            else:
                self._device = torch.device('cpu')
        return self._device

    def load_models(self, model_dir: str = MODEL_DIR) -> bool:
        """Load all available CORAL models"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return False

        if self._loaded:
            return True

        device = self._get_device()
        logger.info(f"Loading CORAL models on {device}")

        # Model files
        model_files = {
            'gait': 'gait_coral_raw_kfold_best.pth',
            'finger_tapping': 'finger_coral_raw_kfold_best.pth',
            'hand_movement': 'hand_coral_raw_kfold_best.pth',
            'leg_agility': 'leg_coral_raw_kfold_best.pth',
        }

        for task, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=device, weights_only=False)
                    config = checkpoint['config']

                    model = MambaCoralModel(
                        input_size=config['input_size'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        num_classes=config['num_classes'],
                        dropout=config['dropout']
                    ).to(device)

                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()

                    self._models[task] = model
                    self._configs[task] = config
                    logger.info(f"Loaded {task} model: input_size={config['input_size']}")

                except Exception as e:
                    logger.error(f"Failed to load {task} model: {e}")
            else:
                logger.warning(f"Model not found: {path}")

        self._loaded = len(self._models) > 0
        logger.info(f"Loaded {len(self._models)} CORAL models")
        return self._loaded

    def is_loaded(self) -> bool:
        return self._loaded

    def get_available_tasks(self) -> List[str]:
        return list(self._models.keys())

    def predict(self, skeleton_sequence: np.ndarray, task: str) -> Optional[CORALPrediction]:
        """
        Predict UPDRS score from skeleton sequence

        Args:
            skeleton_sequence: numpy array of shape (T, F) or (B, T, F)
                - T: number of frames (will be resampled if needed)
                - F: number of features per frame
            task: one of 'gait', 'finger_tapping', 'hand_movement', 'leg_agility'

        Returns:
            CORALPrediction with score 0-4
        """
        if not TORCH_AVAILABLE:
            return None

        if not self._loaded:
            self.load_models()

        if task not in self._models:
            logger.warning(f"No model loaded for task: {task}")
            return None

        model = self._models[task]
        config = self._configs[task]
        device = self._get_device()

        try:
            # Ensure correct shape
            if skeleton_sequence.ndim == 2:
                skeleton_sequence = skeleton_sequence[np.newaxis, ...]  # Add batch dim

            # Resample if needed (task-specific target frames)
            target_frames = {
                'gait': 300,
                'finger_tapping': 150,
                'hand_movement': 150,
                'leg_agility': 150,
            }.get(task, 150)

            if skeleton_sequence.shape[1] != target_frames:
                skeleton_sequence = self._resample(skeleton_sequence, target_frames)

            # Convert to tensor
            x = torch.FloatTensor(skeleton_sequence).to(device)

            # Inference
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)

                # CORAL decoding
                predictions = (probs > 0.5).sum(dim=1)
                expected = probs.sum(dim=1)

                score = int(predictions[0].item())
                expected_score = float(expected[0].item())

                # Calculate confidence from probabilities
                prob_list = probs[0].cpu().numpy().tolist()
                confidence = self._calculate_confidence(prob_list, score)

            return CORALPrediction(
                score=score,
                confidence=round(confidence, 3),
                expected_score=round(expected_score, 3),
                probabilities=[round(p, 3) for p in prob_list],
                task=task,
                model_type="coral_mamba_kfold"
            )

        except Exception as e:
            logger.error(f"Prediction failed for {task}: {e}")
            return None

    def _resample(self, sequence: np.ndarray, target_frames: int) -> np.ndarray:
        """Resample sequence to target number of frames using linear interpolation"""
        from scipy.interpolate import interp1d

        B, T, F = sequence.shape
        if T == target_frames:
            return sequence

        resampled = np.zeros((B, target_frames, F), dtype=np.float32)
        old_indices = np.linspace(0, 1, T)
        new_indices = np.linspace(0, 1, target_frames)

        for b in range(B):
            for f in range(F):
                interp_func = interp1d(old_indices, sequence[b, :, f], kind='linear')
                resampled[b, :, f] = interp_func(new_indices)

        return resampled

    def _calculate_confidence(self, probs: List[float], score: int) -> float:
        """Calculate confidence based on CORAL probability margins"""
        if score == 0:
            return 1 - probs[0] if probs else 0.5
        elif score == 4:
            return probs[-1] if probs else 0.5
        else:
            # Confidence based on margin from 0.5 threshold
            margins = [abs(p - 0.5) for p in probs]
            return min(1.0, 0.5 + np.mean(margins))


def get_coral_scorer() -> CORALScorer:
    """Get singleton CORAL scorer instance"""
    return CORALScorer()
