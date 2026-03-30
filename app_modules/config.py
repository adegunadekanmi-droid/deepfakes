from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
TEMP_DIR = BASE_DIR / "temp"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FRAME_COUNT = 8
FRAME_SIZE = 224
DEVICE = "cpu"

MAX_UPLOAD_MB = 250
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Hugging Face model settings
HF_REPO_ID = "kanmiade/deepfake-detector-model"
HF_FILENAME = "deepfake_resnet152.pt"

# Optional: use token for private repos
HF_TOKEN = os.getenv("HF_TOKEN", None)

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]
