from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "deepfake_resnet152.pt"
REPORT_DIR = BASE_DIR / "reports"
TEMP_DIR = BASE_DIR / "temp"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FRAME_COUNT = 8
FRAME_SIZE = 224
DEVICE = "cpu"

MAX_UPLOAD_MB = 150
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]