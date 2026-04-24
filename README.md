# Automated Player Tracking and Basic Tactical Analysis in Football Videos

**IE 7615 — Neural Networks and Deep Learning | Final Project**
**College of Engineering, Northeastern University**

Charles Appiah · Frank Amankwah · Group 2

> End-to-end deep learning pipeline for player detection, multi-object tracking,
> and team formation classification from broadcast football video.
> Built on **TensorFlow 2.19**, **Keras**, and **Ultralytics YOLOv8**.
> Trained on **SoccerNet EPL 2014–2016** and **StatsBomb Open Data**.

---

## Results

| Stage | Method | Metric | Target | Achieved |
|---|---|---|---|---|
| Detection — Phase 1 baseline | YOLOv8s, COCO pretrained | mAP@0.5 | — | 0.0027 |
| Detection — Phase 2 fine-tuned | YOLOv8s on 33,405 frames | mAP@0.5 | > 0.75 | **0.9010** |
| Tracking | MaskRCNN boxes + IoU tracker | Avg det/frame | > 5 | **9.4** |
| Tactical | CNN on position heatmaps | Test accuracy | > 75% | **98.95%** |

Fine-tuning improvement: +0.8983 mAP@0.5 over the COCO pretrained baseline.

---

## Detection Classes (4 total)

| ID | Class |
|---|---|
| 0 | ball |
| 1 | goalkeeper |
| 2 | player |
| 3 | referee |

---

## System Pipeline

```
SoccerNet .mkv videos (224p, 50 matches × 2 halves)
        │
        ▼
[ preprocess_data.py ]
  Extract frames at 2 FPS using OpenCV → 577,713 .jpg frames
  Convert MaskRCNN JSON boxes → YOLO .txt labels (class 0 = player)
        │
        ▼
[ data_pipeline.py ]
  Split 100 match folders → train/val/test (70/15/15)
  Save split_manifest.json
  Create tf.data.Dataset (640×640, normalized [0,1])
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
[ train_detection.py ]                  [ tactical_model.py ]
  Phase 1: Baseline eval (COCO)           Build heatmap dataset
  Phase 2: Fine-tune YOLOv8s               from StatsBomb lineups
  mAP@0.5: 0.0027 → 0.9010               Train CNN classifier
                                          98.95% test accuracy
        │
        ▼
[ simple_tracker.py ]
  Approach A: model.track() + ByteTrack (Cell 58)
  Approach B: MaskRCNN boxes + SimpleTracker → 9.4 det/frame (Cell 61)
  Approach C: Demo video generator → tracking_demo.mp4 (Cell 79)
        │
        ▼
[ evaluate_system.py ]
  Compare baseline vs fine-tuned
  Full evaluation report + charts
  complete_system_results.json
```

---

## Repository Structure

```
DL-FOOTBALL-TRACKING/
│
├── README.md                    ← this file
├── requirements.txt             ← pip dependencies (matches Cell 1)
│
├── setup_and_paths.py           ← project folder creation + path constants
├── preprocess_data.py           ← frame extraction + MaskRCNN→YOLO conversion
├── data_pipeline.py             ← train/val/test split + tf.data datasets
├── train_detection.py           ← YOLOv8 baseline + fine-tuning + evaluation
├── simple_tracker.py            ← SimpleTracker class + all 3 tracking approaches
├── tactical_model.py            ← heatmap builder + CNN classifier training
├── evaluate_system.py           ← complete system evaluation + charts
│
└── Untitled12__16_.ipynb        ← MAIN NOTEBOOK (all training runs on Colab)
```

> All training was run in the notebook on Google Colab Pro (Tesla T4 GPU).
> The `.py` files are clean, commented extractions of the notebook's core logic
> for code review and reproducibility.

---

## Module Descriptions

### `setup_and_paths.py`
Defines all project path constants (`BASE`, `SOCCERNET`, `STATSBOMB`, etc.)
and `create_project_folders()` to set up the Google Drive directory structure.
Mirrors Cell 1 and Cell 3 of the notebook.

### `preprocess_data.py`
Two main functions:
- `extract_all_frames()` — extracts .jpg frames from SoccerNet .mkv files at 2 FPS
  using OpenCV (Cell 15). Skips if frames already exist.
- `convert_all_bbox_files()` — converts SoccerNet MaskRCNN JSON annotations to
  YOLO .txt format (Cell 21). Each player box is written as class 0.

Note: MaskRCNN annotations only label players (class 0). The 4-class model
(ball/goalkeeper/player/referee) was trained using the Roboflow broadcast dataset.

### `data_pipeline.py`
- `split_match_folders()` — splits 100 match folders 70/15/15 with `random.seed(42)`.
  Splits at folder level to prevent data leakage. Saves `split_manifest.json`.
- `create_tf_dataset()` — builds batched, prefetched `tf.data.Dataset` from folder list.
  Images resized to 640×640 and normalized to [0, 1] (Cell 35).

### `train_detection.py`
- `create_roboflow_yaml()` — writes YAML config for the 4-class Roboflow dataset.
- `run_baseline_evaluation()` — Phase 1: evaluates COCO-pretrained YOLOv8s (Cell 51).
- `run_finetuning()` — Phase 2: fine-tunes on 33,405 SoccerNet frames (Cell 54).
- `train_broadcast_model()` — trains on 298 broadcast + 1,000 SoccerNet images (Cell 49).
- `evaluate_saved_model()` — loads best.pt and evaluates on test set (Cell 53).

### `simple_tracker.py`
The actual tracker used in the project. Three approaches:
- `SimpleTracker` class — IoU-based tracker (Cells 61 and 79). Matches detections
  to tracks frame-by-frame using IoU threshold. Removes stale tracks after N missed frames.
- `track_with_bytetrack()` — runs `model.track(tracker='bytetrack.yaml')` with 3× upscaling
  for 224p SoccerNet frames (Cell 58).
- `process_match_with_maskrcnn()` — uses pre-computed SoccerNet MaskRCNN boxes +
  SimpleTracker. Produced the **9.4 avg detections/frame** result (Cell 61).
- `generate_tracking_video()` — generates `tracking_demo.mp4` using YOLO + SimpleTracker (Cell 79).

### `tactical_model.py`
- `build_heatmap_dataset()` — parses StatsBomb Starting XI events, maps position IDs
  to pitch coordinates via `POSITION_COORDS`, generates 80×120 Gaussian heatmaps (Cell 67).
- `build_tactical_cnn()` — builds the exact CNN architecture trained in Cell 71.
- `train_tactical_model()` — full training pipeline: normalize, split, class weights,
  train with EarlyStopping + ReduceLROnPlateau, save best checkpoint (Cell 71).

### `evaluate_system.py`
- `compare_detection_results()` — loads metric JSONs and prints baseline vs fine-tuned
  comparison (Cell 56).
- `generate_evaluation_report()` — full system report with 3-panel chart (Cell 75).

---

## Datasets

### SoccerNet — Frames + MaskRCNN annotations
- 100 EPL matches (2014–2016), 577,713 extracted frames
- Pre-computed MaskRCNN bounding box JSON files (`*_player_boundingbox_maskrcnn.json`)
- Access: [https://www.soccer-net.org/data](https://www.soccer-net.org/data)
  — password required (`s0cc3rn3t`) for annotation download

```python
from SoccerNet.Downloader import SoccerNetDownloader
dl = SoccerNetDownloader(LocalDirectory='data/soccernet')
dl.password = 's0cc3rn3t'
dl.downloadGame(files=['1_player_boundingbox_maskrcnn.json'], game='2015-02-21 - ...')
```

### Roboflow Football Detection — 4-class broadcast images
- 372 YOLOv8-annotated broadcast images (298 train / 49 val / 25 test)
- 4 classes: ball=0, goalkeeper=1, player=2, referee=3
- Access: [roboflow.com](https://roboflow.com) → search "Football Player Detection" → YOLOv8 format

### StatsBomb Open Data — Formation labels
- 3,464 lineup JSON files, 50 matches with Starting XI formations
- 6,332 heatmap samples extracted for tactical model training
- Access: git clone (free, no registration):

```bash
git clone https://github.com/statsbomb/open-data.git data/statsbomb
```

### Trained Models — Google Drive Download

> **Download link:** [Click here to access all trained model weights](https://drive.google.com/drive/folders/19D7QgmqOYvP6Kfmth_2obXcNY0QYAkJM?usp=sharing)
>
> The Drive folder contains:
> | File | Description | Result |
> |---|---|---|
> | `phase2_finetuned/weights/best.pt` | Main detection model | mAP@0.5 = 0.9010 |
> | `broadcast_combined/weights/best.pt` | Broadcast demo model | mAP@0.5 = 0.8230 |
> | `tactical_cnn_best.keras` | Tactical CNN classifier | 98.95% test accuracy |
> | `phase1_baseline_metrics.json` | Baseline evaluation metrics | mAP@0.5 = 0.0027 |
> | `phase3_finetuned_metrics.json` | Fine-tuned evaluation metrics | mAP@0.5 = 0.9010 |
> | `tactical_metrics.json` | Tactical model metrics | 98.95% accuracy |
> | `complete_system_results.json` | Full system evaluation results | All stages |

To use a model locally, download `best.pt` and run:
```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict("your_frame.jpg", conf=0.15)
```

### Expected folder structure on Google Drive

```
/content/drive/MyDrive/football_project/
├── data/
│   ├── soccernet/              ← SoccerNet match folders + MaskRCNN JSONs
│   ├── statsbomb/              ← StatsBomb open-data clone
│   ├── football_roboflow_clean/ ← Roboflow 4-class broadcast dataset
│   ├── processed/
│   │   ├── images/             ← Extracted .jpg frames (one subfolder per match half)
│   │   └── labels/             ← YOLO .txt label files
│   ├── X_heatmaps.npy          ← Formation heatmaps (6332, 80, 120)
│   ├── y_labels.npy            ← Formation labels (6332,)
│   └── split_manifest.json     ← Train/val/test folder split
└── models/
    ├── phase2_finetuned/weights/best.pt      ← Main detection model (mAP@0.5 = 0.9010)
    ├── broadcast_combined/weights/best.pt    ← Broadcast demo model (mAP@0.5 = 0.823)
    └── tactical_cnn_best.keras               ← Tactical CNN (98.95% accuracy)
```

---

## Installation

**Python 3.10 | Google Colab Pro (Tesla T4)**

```bash
pip install SoccerNet ultralytics opencv-python-headless tensorflow \
            numpy matplotlib scikit-learn pandas Pillow -q
```

Or from requirements.txt:
```bash
pip install -r requirements.txt
```

---

## How to Reproduce

All training was run from the main notebook on Colab. To reproduce:

**1. Run Cell 1** — Mount Drive, install libraries, create folders.

**2. Run Cell 5** — Download SoccerNet MaskRCNN annotation JSONs.

**3. Run Cell 7** — Clone StatsBomb open data.

**4. Run Cell 15** — Extract frames from SoccerNet videos at 2 FPS.

**5. Run Cell 21** — Convert MaskRCNN boxes to YOLO format.

**6. Run Cell 31** — Split match folders 70/15/15, save split_manifest.json.

**7. Run Cell 39** — Verify Roboflow dataset and create YAML config.

**8. Run Cell 51** — Phase 1 baseline evaluation.

**9. Run Cell 54** — Phase 2 fine-tuning on 33,405 frames.

**10. Run Cell 61** — MaskRCNN + SimpleTracker on 3 matches → 9.4 det/frame.

**11. Run Cell 67** — Extract StatsBomb heatmaps.

**12. Run Cell 71** — Train tactical CNN → 98.95% accuracy.

**13. Run Cell 75** — Full system evaluation report.

**14. Run Cell 79** — Generate tracking_demo.mp4.

---

## Key Design Decisions

**Why 3× upscaling before detection?**
SoccerNet frames are 398×224 px (224p). Player bounding boxes are ~10–20 px tall
at native resolution — too small for reliable YOLOv8 detection. Upscaling 3× to
~1194×672 before inference significantly improves recall (Cell 58).

**Why SimpleTracker instead of DeepSORT?**
For the MaskRCNN-based path (Cell 61), boxes are pre-computed and already clean.
IoU matching is sufficient and much faster. For the live YOLO path (Cell 58),
Ultralytics' built-in ByteTrack was used via `model.track(tracker='bytetrack.yaml')`.

**Why split at folder level, not frame level?**
Each folder contains all frames from one match half. Frame-level splitting would
put frames from the same match in both train and test, causing data leakage.

**Why class-weighted loss for the tactical model?**
The 3-5-2 formation has only 455 samples vs 2,107 for 4-2-3-1. Class weights
ensure the minority formation contributes proportionally — it achieved 100% test accuracy.

---

## Hardware

| Item | Value |
|---|---|
| Platform | Google Colab Pro |
| GPU | Tesla T4 (14.9 GB VRAM) |
| TensorFlow | 2.19.0 |
| Ultralytics YOLOv8 | 8.4.37 |
| PyTorch | 2.10.0+cu128 |

---

## Authors

| Name | Email | Contribution |
|---|---|---|
| Charles Appiah | appiah.ch@northeastern.edu | 50% |
| Frank Amankwah | amankwah.f@northeastern.edu | 50% |

GitHub: https://github.com/frankamankwah1zz/DL-FOOTBALL-TRACKING

---

## References

- Redmon & Farhadi (2018). YOLOv3: An Incremental Improvement.
- Wojke et al. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. ICIP.
- Zhang et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV.
- Deliège et al. (2021). SoccerNet-v2. CVPR Workshops.
- StatsBomb Open Data. https://github.com/statsbomb/open-data
- Jocher et al. (2023). Ultralytics YOLO. https://github.com/ultralytics/ultralytics

---

## Generative AI Disclosure

Generative AI tools (Claude) were used for grammar refinement, code debugging,
and report structuring. All technical content, model architectures, experimental
results, and interpretations are the original work of the authors.
