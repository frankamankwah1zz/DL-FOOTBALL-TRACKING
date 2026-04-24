"""
train_detection.py
==================
YOLOv8 detection model training and evaluation.

Extracted from:
  Cell 39  — Load Roboflow dataset and create YAML config
  Cell 41  — Week 5-7 setup (environment check)
  Cell 49  — Train broadcast + Roboflow combined model
  Cell 51  — Phase 1: Baseline evaluation (COCO pretrained, no fine-tuning)
  Cell 54  — Phase 2: Fine-tune YOLOv8s on 33,405 SoccerNet football frames

Classes (4 total, matching the YAML and Roboflow dataset):
    0 = ball
    1 = goalkeeper
    2 = player
    3 = referee

Results achieved:
    Phase 1 baseline (COCO pretrained):  mAP@0.5 = 0.0027
    Phase 2 fine-tuned (33,405 frames):  mAP@0.5 = 0.9010  (+0.8983)
    Broadcast combined model:            mAP@0.5 = 0.8230

Dependencies: ultralytics, torch, yaml, json
"""

import os
import json
import yaml
from pathlib import Path


# ── Project paths ──────────────────────────────────────────────
BASE       = '/content/drive/MyDrive/football_project'
MODELS_DIR = BASE + '/models'
ROBOFLOW   = BASE + '/data/football_roboflow_clean'

# YAML config is written to /content/ (local Colab storage, not Drive)
YAML_CLEAN    = '/content/football_clean.yaml'
YAML_COMBINED = '/content/combined_broadcast.yaml'

# 4 classes used throughout the project
CLASS_NAMES = ['ball', 'goalkeeper', 'player', 'referee']


# ============================================================
# YAML Config Creation (Cell 39)
# ============================================================

def create_roboflow_yaml(roboflow_dir=None, yaml_path=None):
    """
    Create the YAML dataset config for the clean Roboflow dataset.

    The Roboflow dataset has 4 classes: ball, goalkeeper, player, referee.
    This YAML is used for Phase 2 fine-tuning on the full SoccerNet dataset.

    Args:
        roboflow_dir (str): Path to the Roboflow dataset root on Drive.
        yaml_path    (str): Where to write the .yaml file.
    """
    roboflow_dir = roboflow_dir or ROBOFLOW
    yaml_path    = yaml_path    or YAML_CLEAN

    config = {
        'path' : roboflow_dir,
        'train': 'train/images',
        'val'  : 'valid/images',
        'test' : 'test/images',
        'nc'   : 4,
        'names': CLASS_NAMES,   # ['ball', 'goalkeeper', 'player', 'referee']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"YAML config written to: {yaml_path}")
    return yaml_path


def create_combined_yaml(combined_dir='/content/combined_broadcast', yaml_path=None):
    """
    Create the YAML dataset config for the combined broadcast + Roboflow dataset.
    Used for training the broadcast visualization model (Cell 49).

    Args:
        combined_dir (str): Path to combined dataset on Colab local storage.
        yaml_path    (str): Where to write the .yaml file.
    """
    yaml_path = yaml_path or YAML_COMBINED
    config = {
        'path' : combined_dir,
        'train': 'train/images',
        'val'  : 'valid/images',
        'test' : 'test/images',
        'nc'   : 4,
        'names': CLASS_NAMES,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Combined YAML config written to: {yaml_path}")
    return yaml_path


# ============================================================
# Phase 1: Baseline Evaluation (Cell 51)
# ============================================================

def run_baseline_evaluation(yaml_path=None, models_dir=None):
    """
    Evaluate pretrained YOLOv8s (COCO weights) on the football test set.
    No fine-tuning — this is the zero-shot baseline.

    Results: mAP@0.5 = 0.0027 (near zero — COCO weights don't transfer to football)

    Args:
        yaml_path  (str): Path to dataset YAML config.
        models_dir (str): Directory to save evaluation results.

    Returns:
        dict: Baseline metrics (mAP50, mAP50_95, precision, recall).
    """
    from ultralytics import YOLO

    yaml_path  = yaml_path  or YAML_CLEAN
    models_dir = models_dir or MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 50)
    print("PHASE 1: BASELINE EVALUATION")
    print("Pretrained YOLOv8s — COCO weights, no fine-tuning")
    print("=" * 50)

    baseline_model   = YOLO('yolov8s.pt')
    baseline_results = baseline_model.val(
        data    = yaml_path,
        split   = 'test',
        imgsz   = 640,
        batch   = 16,
        device  = 0,
        verbose = True,
        project = models_dir,
        name    = 'phase1_baseline',
    )

    metrics = {
        'phase'    : 'baseline',
        'model'    : 'yolov8s.pt (COCO pretrained)',
        'mAP50'    : float(baseline_results.box.map50),
        'mAP50_95' : float(baseline_results.box.map),
        'precision': float(baseline_results.box.mp),
        'recall'   : float(baseline_results.box.mr),
    }

    print(f"\nPHASE 1 RESULTS")
    print(f"  mAP@0.5      : {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics['mAP50_95']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")

    save_path = os.path.join(models_dir, 'phase1_baseline_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBaseline metrics saved to: {save_path}")
    return metrics


# ============================================================
# Phase 2: Fine-Tuning (Cell 54)
# ============================================================

def run_finetuning(yaml_path=None, models_dir=None):
    """
    Fine-tune YOLOv8s on 33,405 annotated football frames from SoccerNet.

    Training config (matching Cell 54):
        epochs   = 50
        imgsz    = 640
        batch    = 16
        lr0      = 0.001
        patience = 10 (early stopping)
        device   = 0  (GPU)

    This is the model that achieved mAP@0.5 = 0.9010 on the test set.

    Args:
        yaml_path  (str): Path to dataset YAML. Defaults to YAML_CLEAN.
        models_dir (str): Where to save model checkpoints.

    Returns:
        str: Path to best.pt model weights.
    """
    from ultralytics import YOLO

    yaml_path  = yaml_path  or YAML_CLEAN
    models_dir = models_dir or MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 50)
    print("PHASE 2: FINE-TUNING")
    print("Training YOLOv8s on 33,405 football frames")
    print("4 classes: ball, goalkeeper, player, referee")
    print("=" * 50)

    model = YOLO('yolov8s.pt')   # start from COCO pretrained weights

    model.train(
        data     = yaml_path,
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,
        lr0      = 0.001,
        patience = 10,       # early stopping if no improvement
        device   = 0,        # Tesla T4 GPU on Colab Pro
        project  = models_dir,
        name     = 'phase2_finetuned',
        save     = True,
        plots    = True,     # saves training curves to plots/
        verbose  = True,
    )

    best_model_path = os.path.join(models_dir, 'phase2_finetuned/weights/best.pt')
    print(f"\nFine-tuning complete!")
    print(f"Best model: {best_model_path}")
    return best_model_path


# ============================================================
# Broadcast Combined Model (Cell 49)
# ============================================================

def train_broadcast_model(yaml_path=None, models_dir=None):
    """
    Train a broadcast-optimized YOLOv8s model on 298 broadcast images
    + 1,000 SoccerNet frames (combined_broadcast dataset).

    This model achieves mAP@0.5 = 0.823 on full-field broadcast footage
    and was used for poster visualization. It detects 15+ players per frame.

    Training config (matching Cell 49):
        epochs        = 100
        batch         = 16
        lr0           = 0.01
        warmup_epochs = 3
        patience      = 15
        augmentations : mosaic=1.0, fliplr=0.5, scale=0.5, hsv

    Args:
        yaml_path  (str): Path to combined_broadcast.yaml.
        models_dir (str): Where to save model checkpoints.
    """
    import torch
    torch.cuda.empty_cache()

    from ultralytics import YOLO

    yaml_path  = yaml_path  or YAML_COMBINED
    models_dir = models_dir or MODELS_DIR

    print("=" * 50)
    print("TRAINING: Broadcast + Clean Roboflow")
    print("Train: 1,298  |  Valid: 49  |  Test: 25")
    print("=" * 50)

    model = YOLO('yolov8s.pt')
    model.train(
        data          = yaml_path,
        epochs        = 100,
        imgsz         = 640,
        batch         = 16,
        lr0           = 0.01,
        lrf           = 0.01,
        warmup_epochs = 3,
        optimizer     = 'auto',
        patience      = 15,
        save          = True,
        save_period   = 10,
        project       = models_dir,
        name          = 'broadcast_combined',
        exist_ok      = True,
        verbose       = True,
        device        = 0,
        workers       = 4,
        mosaic        = 1.0,
        fliplr        = 0.5,
        scale         = 0.5,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        plots         = True,
    )

    best_path = os.path.join(models_dir, 'broadcast_combined/weights/best.pt')
    print(f"\nTraining complete! Best model: {best_path}")


# ============================================================
# Load Saved Model and Evaluate (Cell 53)
# ============================================================

def evaluate_saved_model(model_path=None, yaml_path=None, models_dir=None):
    """
    Load a saved best.pt checkpoint and evaluate on the test set.
    Compares against Phase 1 baseline if phase1_baseline_metrics.json exists.

    Args:
        model_path (str): Path to best.pt. Searches common locations if None.
        yaml_path  (str): Path to dataset YAML.
        models_dir (str): Directory containing metrics JSON files.

    Returns:
        dict: Evaluation metrics.
    """
    from ultralytics import YOLO

    yaml_path  = yaml_path  or YAML_CLEAN
    models_dir = models_dir or MODELS_DIR

    # Search for saved model
    if model_path is None:
        candidates = [
            os.path.join(models_dir, 'phase2_finetuned2/weights/best.pt'),
            os.path.join(models_dir, 'phase2_finetuned/weights/best.pt'),
        ]
        for p in candidates:
            if Path(p).exists():
                model_path = p
                break

    if model_path is None or not Path(model_path).exists():
        print("No saved model found. Run run_finetuning() first.")
        return None

    print(f"Loading model: {model_path}")
    model   = YOLO(model_path)
    results = model.val(data=yaml_path, split='test', imgsz=640, batch=16,
                        device=0, verbose=True)

    metrics = {
        'phase'    : 'phase2_finetuned',
        'model'    : model_path,
        'mAP50'    : float(results.box.map50),
        'mAP50_95' : float(results.box.map),
        'precision': float(results.box.mp),
        'recall'   : float(results.box.mr),
    }

    print(f"\nPHASE 2 RESULTS — Fine-tuned model")
    print(f"  mAP@0.5      : {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics['mAP50_95']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")

    # Compare against baseline
    baseline_path = os.path.join(models_dir, 'phase1_baseline_metrics.json')
    if Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"\nIMPROVEMENT OVER BASELINE")
        for m in ['mAP50', 'precision', 'recall']:
            b  = baseline[m]
            ft = metrics[m]
            print(f"  {m:<12}: {b:.4f} → {ft:.4f}  (+{ft-b:.4f})")

    save_path = os.path.join(models_dir, 'phase2_finetuned_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {save_path}")
    return metrics


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("Detection Model Training")
    print("=" * 50)
    print("1. Create YAML config")
    create_roboflow_yaml()
    print("\n2. Phase 1: Baseline (COCO pretrained, no fine-tuning)")
    run_baseline_evaluation()
    print("\n3. Phase 2: Fine-tune on SoccerNet data")
    run_finetuning()
    print("\n4. Evaluate fine-tuned model")
    evaluate_saved_model()
