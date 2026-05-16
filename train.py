"""
Train the custom face classifier on processed face crops.

Uses paths from config/paths.py (.env / FACE_TRAIN_PROCESSED_DIR for SSD splits).

Run:
  python train.py
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "src" / "04_train_model.py"),
        run_name="__main__",
    )
