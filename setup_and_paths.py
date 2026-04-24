"""
setup_and_paths.py
==================
Project setup and path definitions.
Extracted from Cell 1 (Full Project Setup) and Cell 3 (Path Definitions).

Run this first before any other script to mount Drive and verify all paths.
On Google Colab, run Cell 1 of the notebook instead.
"""

import os
from pathlib import Path

# ── Project base path (Google Drive) ──────────────────────────
BASE = '/content/drive/MyDrive/football_project'

# ── All project paths ──────────────────────────────────────────
SOCCERNET  = os.path.join(BASE, 'data/soccernet')
STATSBOMB  = os.path.join(BASE, 'data/statsbomb')
PROCESSED  = os.path.join(BASE, 'data/processed')
FRAMES_DIR = os.path.join(PROCESSED, 'images')
LABELS_DIR = os.path.join(PROCESSED, 'labels')
TRAIN      = os.path.join(BASE, 'data/train')
VAL        = os.path.join(BASE, 'data/val')
TEST       = os.path.join(BASE, 'data/test')
MODELS     = os.path.join(BASE, 'models')
RESULTS    = os.path.join(BASE, 'results')
FIGURES    = os.path.join(BASE, 'figures')


def create_project_folders():
    """
    Create all required project directories.
    Mirrors the folder structure created in Cell 1 of the notebook.
    Safe to call multiple times — uses exist_ok=True.
    """
    folders = [
        BASE,
        SOCCERNET,
        STATSBOMB,
        PROCESSED,
        FRAMES_DIR,
        LABELS_DIR,
        TRAIN,
        VAL,
        TEST,
        MODELS,
        RESULTS,
        FIGURES,
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print(f"TensorFlow check:")
    try:
        import tensorflow as tf
        print(f"  TensorFlow version : {tf.__version__}")
        print(f"  GPU available      : {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("  TensorFlow not installed")

    print(f"\nBase folder: {BASE}")
    print("Project setup complete.")


def verify_paths():
    """Print status of all key project paths."""
    print("Path verification:")
    for name, path in {
        'BASE'      : BASE,
        'SOCCERNET' : SOCCERNET,
        'STATSBOMB' : STATSBOMB,
        'FRAMES_DIR': FRAMES_DIR,
        'LABELS_DIR': LABELS_DIR,
        'MODELS'    : MODELS,
    }.items():
        status = "OK" if Path(path).exists() else "MISSING"
        print(f"  [{status}]  {name}: {path}")


if __name__ == "__main__":
    create_project_folders()
    verify_paths()
