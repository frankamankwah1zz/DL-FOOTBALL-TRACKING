"""
data_pipeline.py
================
Data splitting and TensorFlow dataset creation.

Extracted from:
  Cell 31 — Split match folders into train / val / test (70/15/15)
             Saves split_manifest.json. Reproducible via random.seed(42).
  Cell 33 — Count frames per split, generate statistics and charts.
  Cell 35 — Create tf.data.Dataset objects for training.

Split strategy:
  Split is done at the FOLDER level (one folder = one half of one match),
  not the frame level. This prevents data leakage — all frames from the
  same match stay in the same split.

TensorFlow dataset:
  Images are resized to 640×640 and normalized to [0.0, 1.0].
  Datasets are shuffled (train only), batched, and prefetched.

Dependencies: tensorflow, Pillow, matplotlib
"""

import os
import json
import random
from pathlib import Path

import tensorflow as tf


# ── Project paths ──────────────────────────────────────────────
BASE         = '/content/drive/MyDrive/football_project'
FRAMES_DIR   = BASE + '/data/processed/images'
LABELS_DIR   = BASE + '/data/processed/labels'
MANIFEST_PATH = BASE + '/data/split_manifest.json'


# ============================================================
# Train / Val / Test Split (Cell 31)
# ============================================================

def split_match_folders(frames_dir, manifest_path, seed=42):
    """
    Randomly split match frame folders into train / val / test (70/15/15).

    Uses random.seed(42) for reproducibility — the same split is produced
    every time this function runs from scratch.

    Guards against re-running: if split_manifest.json already exists,
    it loads and returns the existing split without re-shuffling.

    Args:
        frames_dir    (str): Root directory of extracted frame folders.
        manifest_path (str): Path to save/load split_manifest.json.
        seed          (int): Random seed. Default 42.

    Returns:
        dict: Manifest with keys train_dirs, val_dirs, test_dirs,
              train_label_dirs, val_label_dirs, test_label_dirs.
    """
    # Guard: load existing split if already done
    if Path(manifest_path).exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print("split_manifest.json already exists — loading existing split.")
        print(f"  Train : {len(manifest['train_dirs'])} folders")
        print(f"  Val   : {len(manifest['val_dirs'])} folders")
        print(f"  Test  : {len(manifest['test_dirs'])} folders")
        return manifest

    random.seed(seed)
    all_folders = sorted([f.name for f in Path(frames_dir).iterdir() if f.is_dir()])
    print(f"Total match folders found: {len(all_folders)}")

    random.shuffle(all_folders)
    n       = len(all_folders)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_folders = all_folders[:n_train]
    val_folders   = all_folders[n_train:n_train + n_val]
    test_folders  = all_folders[n_train + n_val:]

    manifest = {
        'train_dirs'      : [str(Path(frames_dir) / f) for f in train_folders],
        'val_dirs'        : [str(Path(frames_dir) / f) for f in val_folders],
        'test_dirs'       : [str(Path(frames_dir) / f) for f in test_folders],
        'train_label_dirs': [str(Path(LABELS_DIR) / f) for f in train_folders],
        'val_label_dirs'  : [str(Path(LABELS_DIR) / f) for f in val_folders],
        'test_label_dirs' : [str(Path(LABELS_DIR) / f) for f in test_folders],
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"split_manifest.json saved.")
    print(f"  Train : {len(train_folders)} folders (~70%)")
    print(f"  Val   : {len(val_folders)} folders (~15%)")
    print(f"  Test  : {len(test_folders)} folders (~15%)")
    return manifest


def load_manifest(manifest_path=None):
    """
    Load split_manifest.json. Run split_match_folders() first if it doesn't exist.

    Returns:
        dict: Manifest with train_dirs, val_dirs, test_dirs keys.
    """
    path = Path(manifest_path or MANIFEST_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"split_manifest.json not found at {path}. "
            "Run split_match_folders() first."
        )
    with open(path) as f:
        return json.load(f)


# ============================================================
# Frame Statistics (Cell 33)
# ============================================================

def count_frames_in_split(folder_list):
    """
    Count total .jpg frames across a list of folder paths.

    Args:
        folder_list (list): List of folder path strings.

    Returns:
        int: Total number of .jpg files found.
    """
    total = 0
    for folder in folder_list:
        p = Path(folder)
        if p.exists():
            total += len(list(p.glob('*.jpg')))
    return total


def print_split_statistics(manifest):
    """
    Print frame counts per split and basic per-folder statistics.

    Args:
        manifest (dict): Loaded split_manifest.json.
    """
    train_count = count_frames_in_split(manifest['train_dirs'])
    val_count   = count_frames_in_split(manifest['val_dirs'])
    test_count  = count_frames_in_split(manifest['test_dirs'])
    total       = train_count + val_count + test_count

    print("=" * 50)
    print("DATA SPLIT STATISTICS")
    print("=" * 50)
    if total > 0:
        print(f"  Train      : {train_count:>8,} frames  ({train_count/total*100:.1f}%)"
              f"  —  {len(manifest['train_dirs'])} folders")
        print(f"  Validation : {val_count:>8,} frames  ({val_count/total*100:.1f}%)"
              f"  —  {len(manifest['val_dirs'])} folders")
        print(f"  Test       : {test_count:>8,} frames  ({test_count/total*100:.1f}%)"
              f"  —  {len(manifest['test_dirs'])} folders")
        print(f"  Total      : {total:>8,} frames  (100%)")
    print("=" * 50)


# ============================================================
# TensorFlow Dataset Creation (Cell 35)
# ============================================================

def get_image_paths(folder_list):
    """
    Collect all .jpg paths from a list of folders, sorted.

    Args:
        folder_list (list): List of folder path strings.

    Returns:
        list[str]: Sorted list of .jpg file path strings.
    """
    paths = []
    for folder in folder_list:
        p = Path(folder)
        if p.exists():
            paths.extend(sorted(p.glob('*.jpg')))
    return [str(p) for p in paths]


def preprocess_image(image_path):
    """
    Load a JPEG, resize to 640×640, and normalize to [0.0, 1.0].

    This is the same preprocessing used in Cell 35 of the notebook.
    640×640 matches the YOLOv8 training input resolution.

    Args:
        image_path (tf.Tensor): String tensor with the image file path.

    Returns:
        tf.Tensor: Float32 tensor of shape (640, 640, 3), range [0, 1].
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [640, 640])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def create_tf_dataset(folder_list, batch_size=32, shuffle=True, seed=42):
    """
    Create a batched, prefetched tf.data.Dataset from a list of frame folders.

    Exactly matches the dataset created in Cell 35 of the notebook.

    Args:
        folder_list (list): List of folder path strings (from manifest).
        batch_size  (int):  Samples per batch. Default 32.
        shuffle     (bool): Shuffle with buffer_size=1000. Default True.
        seed        (int):  Shuffle seed. Default 42.

    Returns:
        tuple: (tf.data.Dataset, int) — dataset and total image count.
    """
    image_paths = get_image_paths(folder_list)
    print(f"  Images found: {len(image_paths):,}")

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, len(image_paths)


def create_all_datasets(manifest=None, batch_size=32):
    """
    Create train, val, and test tf.data.Dataset objects from the manifest.

    Args:
        manifest   (dict | None): Loaded manifest, or None to load from disk.
        batch_size (int):         Batch size. Default 32.

    Returns:
        tuple: (train_ds, val_ds, test_ds) — three tf.data.Dataset objects.
    """
    if manifest is None:
        manifest = load_manifest()

    print("Creating TensorFlow datasets...")
    print("\nTrain dataset:")
    train_ds, train_n = create_tf_dataset(manifest['train_dirs'], batch_size, shuffle=True)
    print("Validation dataset:")
    val_ds,   val_n   = create_tf_dataset(manifest['val_dirs'],   batch_size, shuffle=False)
    print("Test dataset:")
    test_ds,  test_n  = create_tf_dataset(manifest['test_dirs'],  batch_size, shuffle=False)

    print(f"\n{'='*50}")
    print("TENSORFLOW DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Image size  : 640 × 640 × 3")
    print(f"  Pixel range : [0.0, 1.0] (normalized)")
    print(f"  Batch size  : {batch_size}")
    print(f"  Train       : {train_n:,} images  ({train_n // batch_size:,} batches)")
    print(f"  Val         : {val_n:,} images  ({val_n // batch_size:,} batches)")
    print(f"  Test        : {test_n:,} images  ({test_n // batch_size:,} batches)")
    print(f"{'='*50}")

    return train_ds, val_ds, test_ds


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("Data Pipeline")
    print("=" * 50)

    manifest = split_match_folders(FRAMES_DIR, MANIFEST_PATH)
    print_split_statistics(manifest)

    print("\nCreating TensorFlow datasets...")
    train_ds, val_ds, test_ds = create_all_datasets(manifest)

    # Verify by pulling one batch
    print("\nVerifying datasets (one batch each):")
    for name, ds in [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]:
        batch = next(iter(ds))
        print(f"  {name:<6} | shape: {batch.shape} "
              f"| range: [{batch.numpy().min():.2f}, {batch.numpy().max():.2f}]")

    print("\nDatasets ready for YOLOv8 training.")
