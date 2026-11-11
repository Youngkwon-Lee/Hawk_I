"""Private data locations (override with environment variables). No data is stored in Git."""
import os
DATA_ROOT = os.environ.get("PD4T_TURN_ROOT", "/home/yk/private/turnpose")
CLIP_ROOT = os.environ.get("PD4T_CLIP_ROOT", "/mnt/d/clip/gait")
LABEL_DIR = os.environ.get("PD4T_TURN_LABEL_DIR", "/home/yk/gait_export")
OFFICIAL_DIR = os.environ.get("PD4T_OFFICIAL_ANN", "/mnt/d/1_PD4T/PD4T/PD4T/Annotations/Gait")
