"""
Train the custom face classifier on processed face crops.
Uses importlib to bypass Python's naming restrictions for modules starting with numbers.
"""
import importlib

from project_paths import NOTEBOOKS_ROOT, setup_sys_path

setup_sys_path()

if __name__ == "__main__":
    train_module = importlib.import_module("src.04_train_model")
    train_module.train_model()
