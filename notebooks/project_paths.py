"""
Shared path resolution for scripts under notebooks/.
All data lives at project root: dataset/ and database/
"""
from pathlib import Path

NOTEBOOKS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOKS_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "users_embeddings.pkl"
LOG_FILE_PATH = DATABASE_DIR / "access_log.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def setup_sys_path():
    """Allow imports: config.*, src.* from notebooks/."""
    import sys

    root = str(PROJECT_ROOT)
    nb = str(NOTEBOOKS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if nb not in sys.path:
        sys.path.insert(0, nb)


def ensure_data_dirs():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
