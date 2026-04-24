"""
tactical_model.py
=================
Tactical formation classifier — CNN trained on player-position heatmaps.

Extracted from:
  Cell 67 — Build heatmap dataset from StatsBomb Starting XI events
  Cell 71 — Train CNN tactical classifier (the actual model in the paper)
  Cell 73 — Plot training history curves

The model takes an 80×120 grayscale heatmap of player positions on a
football pitch and classifies it into one of 4 team formations.

4 formation classes:
    0 = 4-3-3   (1,775 training samples)
    1 = 4-4-2   (1,995 training samples)
    2 = 4-2-3-1 (2,107 training samples — most common)
    3 = 3-5-2   (  455 training samples — least common)

Class imbalance is handled with class_weight='balanced' in model.fit().

CNN architecture (Cell 71):
    Input: (80, 120, 1)
    Conv2D(32)→BN→Conv2D(32)→MaxPool(2)→Dropout(0.25)
    Conv2D(64)→BN→Conv2D(64)→MaxPool(2)→Dropout(0.25)
    Conv2D(128)→BN→GlobalAvgPool→Dropout(0.5)
    Dense(128,relu)→Dropout(0.5)→Dense(4, softmax)

Results:
    Test accuracy: 98.95%  (target was 75%)
    Per-class: 4-3-3=98.55%, 4-4-2=97.90%, 4-2-3-1=100%, 3-5-2=100%

Dependencies: tensorflow, keras, scikit-learn, numpy, matplotlib
"""

import json
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


# ── Project paths ──────────────────────────────────────────────
BASE      = '/content/drive/MyDrive/football_project'
STATSBOMB = BASE + '/data/statsbomb'

# Formation class mapping
CLASS_NAMES = {0: '4-3-3', 1: '4-4-2', 2: '4-2-3-1', 3: '3-5-2'}
NUM_CLASSES = 4

# StatsBomb formation integer codes → class indices
# Rare formations are merged into the nearest common class
FORMATION_MAP = {
    433  : 0,    # 4-3-3
    442  : 1,    # 4-4-2
    4231 : 2,    # 4-2-3-1
    41221: 1,    # → 4-4-2
    352  : 3,    # 3-5-2
    4222 : 1,    # → 4-4-2
    41212: 1,    # → 4-4-2
    343  : 0,    # → 4-3-3
    3412 : 3,    # → 3-5-2
    4141 : 1,    # → 4-4-2
    4321 : 2,    # → 4-2-3-1
    4312 : 2,    # → 4-2-3-1
}

# StatsBomb position IDs → approximate (x, y) yard coordinates on a 120×80 pitch
# Used to convert lineup positions to heatmap pixel coordinates
POSITION_COORDS = {
    1 : (5,   40),   # Goalkeeper
    2 : (20,  10),   # Right Back
    3 : (20,  30),   # Right Center Back
    4 : (20,  50),   # Left Center Back
    5 : (20,  70),   # Left Back
    6 : (35,  20),   # Right Wing Back
    7 : (35,  60),   # Left Wing Back
    8 : (45,  20),   # Right Defensive Midfield
    9 : (45,  40),   # Center Defensive Midfield
    10: (45,  60),   # Left Defensive Midfield
    11: (55,  20),   # Right Center Midfield
    12: (55,  40),   # Center Midfield
    13: (55,  60),   # Left Center Midfield
    14: (65,  15),   # Right Midfield
    15: (65,  65),   # Left Midfield
    16: (65,  25),   # Right Attacking Midfield
    17: (65,  40),   # Center Attacking Midfield
    18: (65,  55),   # Left Attacking Midfield
    19: (75,  10),   # Right Wing
    20: (75,  70),   # Left Wing
    21: (80,  25),   # Right Center Forward
    22: (80,  40),   # Center Forward
    23: (80,  55),   # Left Center Forward
    24: (75,  40),   # Secondary Striker
    25: (85,  40),   # Striker
}


# ============================================================
# Heatmap Generation (Cell 67)
# ============================================================

def positions_to_heatmap(players, field_h=80, field_w=120):
    """
    Convert a list of player (x, y) yard coordinates to a 2D Gaussian heatmap.

    Each player contributes a Gaussian blob (radius ~3 cells) centred at
    their grid position. This gives the CNN spatially structured input where
    each formation produces a visually distinct player distribution.

    Args:
        players (list): List of (x, y) coordinates in pitch yards.
        field_h (int):  Heatmap height (pitch height). Default 80.
        field_w (int):  Heatmap width (pitch width).  Default 120.

    Returns:
        np.ndarray: float32 array of shape (field_h, field_w).
    """
    heatmap = np.zeros((field_h, field_w), dtype=np.float32)
    for x, y in players:
        xi = min(int(x), field_w - 1)
        yi = min(int(y), field_h - 1)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = xi + dx, yi + dy
                if 0 <= nx < field_w and 0 <= ny < field_h:
                    heatmap[ny, nx] += np.exp(-(dx**2 + dy**2) / 5.0)
    return heatmap


def build_heatmap_dataset(statsbomb_dir, save_dir=None):
    """
    Extract formation heatmaps from StatsBomb Starting XI events.

    Reads all events/*.json files, finds 'Starting XI' events,
    extracts player positions via POSITION_COORDS, converts to heatmap,
    and maps the formation code to a class label via FORMATION_MAP.

    Saves X_heatmaps.npy (N, 80, 120) and y_labels.npy (N,) to save_dir.

    Args:
        statsbomb_dir (str): Root StatsBomb data directory.
        save_dir      (str): Where to save .npy files. Defaults to BASE/data/.

    Returns:
        tuple: (X, y) — numpy arrays ready for model training.
    """
    save_dir    = save_dir or (BASE + '/data')
    event_files = sorted(Path(statsbomb_dir).rglob('events/*.json'))
    print(f"Scanning {len(event_files)} event files...")

    X_heatmaps = []
    y_labels   = []
    match_info = []

    for event_file in event_files:
        with open(event_file) as f:
            events = json.load(f)

        for event in events:
            if event.get('type', {}).get('name') != 'Starting XI':
                continue

            tactics   = event.get('tactics', {})
            formation = tactics.get('formation')
            lineup    = tactics.get('lineup', [])
            team      = event.get('team', {}).get('name', 'Unknown')

            if not formation or formation not in FORMATION_MAP:
                continue

            label = FORMATION_MAP[formation]

            # Map position IDs to yard coordinates
            positions = []
            for player in lineup:
                pos_id = player.get('position', {}).get('id', 0)
                if pos_id in POSITION_COORDS:
                    positions.append(POSITION_COORDS[pos_id])

            if len(positions) < 10:  # need at least 10 players
                continue

            heatmap = positions_to_heatmap(positions)
            X_heatmaps.append(heatmap)
            y_labels.append(label)
            match_info.append({
                'match_id' : event_file.stem,
                'team'     : team,
                'formation': formation,
                'label'    : label,
                'class'    : CLASS_NAMES[label],
            })

    X = np.array(X_heatmaps)
    y = np.array(y_labels)

    print(f"\nDataset summary:")
    print(f"  Total samples : {len(X)}")
    print(f"  Heatmap shape : {X[0].shape}")
    print(f"\nClass distribution:")
    for label, name in CLASS_NAMES.items():
        count = np.sum(y == label)
        print(f"  {name:<12}: {count} samples")

    np.save(f'{save_dir}/X_heatmaps.npy', X)
    np.save(f'{save_dir}/y_labels.npy',   y)
    print(f"\nSaved X_heatmaps.npy and y_labels.npy to {save_dir}")
    return X, y


# ============================================================
# CNN Model (Cell 71 — exact architecture)
# ============================================================

def build_tactical_cnn(input_shape=(80, 120, 1), num_classes=4):
    """
    Build the tactical CNN classifier.

    This is the EXACT architecture trained in Cell 71 of the notebook
    that achieved 98.95% test accuracy.

    Architecture:
        Block 1: Conv2D(32,3,same,relu) → BN → Conv2D(32,3,same,relu)
                 → MaxPool(2) → Dropout(0.25)
        Block 2: Conv2D(64,3,same,relu) → BN → Conv2D(64,3,same,relu)
                 → MaxPool(2) → Dropout(0.25)
        Block 3: Conv2D(128,3,same,relu) → BN → GlobalAvgPool → Dropout(0.5)
        Head:    Dense(128,relu) → Dropout(0.5) → Dense(4, softmax)

    Args:
        input_shape (tuple): (height, width, channels). Default (80, 120, 1).
        num_classes (int):   Number of formation classes. Default 4.

    Returns:
        keras.Model: Uncompiled model named 'tactical_cnn'.
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # Classification head
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='tactical_cnn')


def train_tactical_model(X, y, models_dir=None):
    """
    Full training pipeline matching Cell 71 exactly.

    Steps:
      1. Normalize heatmaps to [0, 1]
      2. Add channel dim → (N, 80, 120, 1)
      3. One-hot encode labels
      4. Stratified 70/15/15 train/val/test split
      5. Compute class weights (balanced) for imbalanced 3-5-2
      6. Build CNN, compile, train with EarlyStopping + ReduceLROnPlateau
      7. Evaluate on test set, print per-class accuracy
      8. Save model + metrics JSON

    Args:
        X         (np.ndarray): Heatmap array (N, 80, 120).
        y         (np.ndarray): Integer label array (N,).
        models_dir (str):       Where to save the .keras checkpoint.

    Returns:
        tuple: (model, history)
    """
    models_dir = models_dir or (BASE + '/models')

    # 1. Normalize
    X = X / X.max()

    # 2. Add channel dimension
    X = X[..., np.newaxis]   # (N, 80, 120, 1)
    print(f"X shape after reshape: {X.shape}")

    # 3. One-hot encode
    y_cat = keras.utils.to_categorical(y, NUM_CLASSES)

    # 4. Stratified split: 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_cat, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_train):,}")
    print(f"  Val   : {len(X_val):,}")
    print(f"  Test  : {len(X_test):,}")

    # 5. Class weights for imbalanced 3-5-2 class
    y_integers        = np.argmax(y_train, axis=1)
    class_weights_arr = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    print(f"\nClass weights: {class_weight_dict}")

    # 6. Build and compile
    model = build_tactical_cnn()
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Callbacks
    checkpoint_path = f'{models_dir}/tactical_cnn_best.keras'
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor='val_accuracy',
            save_best_only=True, verbose=1,
        ),
    ]

    print("\n" + "="*50)
    print("TRAINING TACTICAL CNN")
    print("="*50)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate on test set
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    y_pred_cl = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true_cl = np.argmax(y_test, axis=1)

    print(f"\nPer-class accuracy:")
    for label, name in CLASS_NAMES.items():
        mask = y_true_cl == label
        if mask.sum() > 0:
            acc = np.mean(y_pred_cl[mask] == label)
            print(f"  {name:<12}: {acc:.4f}  ({mask.sum()} samples)")

    # 8. Save metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss'    : float(test_loss),
        'train_samples': len(X_train),
        'val_samples'  : len(X_val),
        'test_samples' : len(X_test),
        'classes'      : CLASS_NAMES,
    }
    metrics_path = f'{models_dir}/tactical_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved    → {checkpoint_path}")
    print(f"Metrics saved  → {metrics_path}")
    return model, history


def plot_training_history(history):
    """
    Plot accuracy and loss curves from training history (Cell 73).

    Args:
        history: Keras History object returned by model.fit().
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tactical CNN Training History', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='orange')
    axes[0].axhline(y=0.75, color='red', linestyle='--', label='Target (75%)')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', color='blue')
    axes[1].plot(history.history['val_loss'], label='Val',   color='orange')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/content/tactical_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training curves saved.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("Tactical Formation CNN")
    print("=" * 50)

    # Build and display model
    model = build_tactical_cnn()
    model.summary()

    # Verify output shape
    dummy = np.random.rand(4, 80, 120, 1).astype(np.float32)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    out   = model.predict(dummy, verbose=0)
    print(f"\nInput : {dummy.shape}")
    print(f"Output: {out.shape}")   # (4, 4)
    print(f"Probs : {out.sum(axis=1)}")  # all ~1.0

    print("\nTo train: load X_heatmaps.npy + y_labels.npy and call train_tactical_model(X, y)")
