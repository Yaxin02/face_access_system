import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_override = os.getenv("FACE_DATA_ROOT")
if _override:
    DATA_ROOT = Path(_override).expanduser()
else:
    flash = Path(os.getenv("FACE_FLASH_ROOT", "/Volumes/HIKSEMI")).expanduser()
    DATA_ROOT = flash / os.getenv("FACE_DATA_FOLDER", "face_access_data")

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"
SPLITS_DIR = DATA_ROOT / "splits"
EXTERNAL_DIR = DATA_ROOT / "external"
VGGFACE2_DIR = DATA_ROOT / "vggface2"

# Override for training only (e.g. SSD .../train with one folder per identity)
_train_override = os.getenv("FACE_TRAIN_PROCESSED_DIR")
TRAIN_PROCESSED_DIR = (
    Path(_train_override).expanduser() if _train_override else PROCESSED_DIR
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for folder in [
    RAW_DIR,
    PROCESSED_DIR,
    EMBEDDINGS_DIR,
    SPLITS_DIR,
    EXTERNAL_DIR,
    VGGFACE2_DIR,
    MODELS_DIR,
    REPORTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
