"""
preprocess_data.py
==================
Data preprocessing: frame extraction from SoccerNet videos and
conversion of MaskRCNN bounding boxes to YOLO label format.

Extracted from:
  Cell 13 — Setup (paths, folders)
  Cell 15 — Extract frames from .mkv videos at 2 FPS using OpenCV
  Cell 19 — Check frame size (PIL)
  Cell 21 — Convert MaskRCNN JSON bounding boxes to YOLO .txt format

YOLO label format written (one object per line):
    <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].

Class IDs used throughout this project:
    0 = ball
    1 = goalkeeper
    2 = player
    3 = referee

Note: MaskRCNN annotations from SoccerNet label ALL detected people
as class 0 (player). The Roboflow broadcast dataset provides all 4
classes and was used for the fine-tuned model.

Dependencies: opencv-python-headless, Pillow, numpy
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path


# ── Project paths (update BASE to match your Drive) ───────────
BASE       = '/content/drive/MyDrive/football_project'
SOCCERNET  = os.path.join(BASE, 'data/soccernet')
PROCESSED  = os.path.join(BASE, 'data/processed')
FRAMES_DIR = os.path.join(PROCESSED, 'images')
LABELS_DIR = os.path.join(PROCESSED, 'labels')


# ============================================================
# Frame Extraction (Cell 15)
# ============================================================

def extract_frames(video_path, output_dir, fps_extract=2):
    """
    Extract frames from a SoccerNet .mkv video at a given FPS.

    Creates a subfolder named after the match+half inside output_dir.
    Only runs frame extraction — skips if the folder already has .jpg files
    (checked by the caller in Cell 15).

    Args:
        video_path  (str): Path to .mkv video file.
        output_dir  (str): Root directory for frame folders.
        fps_extract (int): Target extraction rate. Default 2 FPS.
                           SoccerNet native rate is 25 FPS → saves every 12th frame.

    Returns:
        tuple: (saved_count, frame_timestamps, save_dir_path)
    """
    video_name = Path(video_path).stem       # e.g. "1_224p"
    match_name = Path(video_path).parent.name # e.g. "2015-02-21 - Chelsea..."
    safe_name  = match_name.replace(' ', '_').replace('/', '_')
    save_dir   = Path(output_dir) / f"{safe_name}_{video_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    cap       = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval  = max(1, int(video_fps / fps_extract))  # e.g. 25/2 = 12

    frame_count  = 0
    saved_count  = 0
    frame_times  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            fname = save_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            frame_times.append(frame_count / video_fps)
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count, frame_times, str(save_dir)


def extract_all_frames(soccernet_dir, frames_dir, fps_extract=2):
    """
    Extract frames from all SoccerNet .mkv videos found recursively.
    Saves a frame_log.json recording frame counts and timestamps per video.

    Args:
        soccernet_dir (str): Root SoccerNet directory containing match folders.
        frames_dir    (str): Output directory for extracted frame folders.
        fps_extract   (int): Extraction FPS. Default 2.
    """
    # Guard: skip if frames already exist
    existing = sum(1 for _ in Path(frames_dir).rglob('*.jpg'))
    if existing > 0:
        print(f"Frames already extracted — {existing:,} .jpg files found. Skipping.")
        return

    all_videos = sorted(Path(soccernet_dir).rglob('*.mkv'))
    print(f"Found {len(all_videos)} .mkv files. Extracting at {fps_extract} FPS...")

    frame_log    = {}
    total_frames = 0

    for i, video_path in enumerate(all_videos):
        print(f"[{i+1}/{len(all_videos)}] {video_path.parent.name}/{video_path.name}")
        count, times, save_dir = extract_frames(str(video_path), frames_dir, fps_extract)
        frame_log[str(video_path)] = {
            'frames'    : count,
            'timestamps': times,
            'save_dir'  : save_dir,
        }
        total_frames += count
        print(f"  {count} frames saved to {Path(save_dir).name}")

    print(f"\nTotal frames extracted: {total_frames:,}")

    log_path = os.path.join(BASE, 'frame_log.json')
    with open(log_path, 'w') as f:
        json.dump(frame_log, f, indent=2)
    print(f"Frame log saved to {log_path}")


# ============================================================
# Frame Size Check (Cell 19)
# ============================================================

def check_frame_size(frames_dir):
    """
    Report the resolution of the first extracted frame found.

    SoccerNet 224p frames are 398 × 224 pixels.
    This small size is why 3× upscaling is used before YOLOv8 detection.

    Args:
        frames_dir (str): Root directory of extracted frame folders.
    """
    from PIL import Image

    sample_frame = next(Path(frames_dir).rglob('*.jpg'), None)
    if sample_frame is None:
        print("No frames found.")
        return

    img = Image.open(sample_frame)
    print(f"Frame size   : {img.size[0]} × {img.size[1]} pixels")
    print(f"Mode         : {img.mode}")
    print(f"Sample file  : {sample_frame.name}")
    print(f"File size    : {sample_frame.stat().st_size / 1024:.1f} KB")
    print(f"\nNote: 398×224 is small. Use upscale_factor=3 before YOLOv8 detection.")


# ============================================================
# MaskRCNN → YOLO Conversion (Cell 21)
# ============================================================

def maskrcnn_to_yolo(bbox, img_width, img_height):
    """
    Convert a MaskRCNN bounding box [x1, y1, x2, y2] (pixels) to
    YOLO format (x_center, y_center, width, height), all normalized to [0,1].

    Clamps values to [0, 1] to handle rare out-of-bound annotations.

    Args:
        bbox       (list): [x1, y1, x2, y2] in pixel coordinates.
        img_width  (int):  Image width in pixels.
        img_height (int):  Image height in pixels.

    Returns:
        tuple: (x_center, y_center, width, height) normalized to [0, 1].
    """
    x1, y1, x2, y2 = bbox
    xc = max(0.0, min(1.0, (x1 + x2) / 2 / img_width))
    yc = max(0.0, min(1.0, (y1 + y2) / 2 / img_height))
    w  = max(0.0, min(1.0, (x2 - x1) / img_width))
    h  = max(0.0, min(1.0, (y2 - y1) / img_height))
    return xc, yc, w, h


def convert_bbox_file(bbox_json_path, labels_dir, class_id=0):
    """
    Convert one SoccerNet MaskRCNN JSON file to YOLO .txt label files.

    SoccerNet MaskRCNN JSON structure:
        {
          "size": [num_frames, height, width, channels],
          "predictions": [
            {"bboxes": [[x1,y1,x2,y2], ...], "onfield": [1,0,1,...]},
            ...
          ]
        }

    One .txt file is created per frame. Empty .txt = no detections.

    Args:
        bbox_json_path (str): Path to *_player_boundingbox_maskrcnn.json.
        labels_dir     (str): Directory to write .txt label files.
        class_id       (int): YOLO class ID to assign. Default 0.
                               Note: SoccerNet only provides player boxes
                               (no ball/goalkeeper/referee annotations).

    Returns:
        int: Number of label files written.
    """
    with open(bbox_json_path, 'r') as f:
        data = json.load(f)

    num_frames, img_height, img_width, _ = data['size']
    predictions = data['predictions']
    print(f"   Image size: {img_width}×{img_height}  |  Frames: {num_frames}")

    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    converted = 0

    for frame_idx, frame_data in enumerate(predictions):
        label_path = Path(labels_dir) / f"frame_{frame_idx:06d}.txt"
        lines = []
        for bbox in frame_data.get('bboxes', []):
            if len(bbox) != 4:
                continue
            xc, yc, w, h = maskrcnn_to_yolo(bbox, img_width, img_height)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        label_path.write_text('\n'.join(lines))
        converted += 1

    return converted


def convert_all_bbox_files(soccernet_dir, labels_dir):
    """
    Convert all SoccerNet MaskRCNN JSON files to YOLO format.
    Skips conversion if .txt label files already exist.

    Args:
        soccernet_dir (str): Root SoccerNet directory.
        labels_dir    (str): Root output directory for YOLO .txt files.
    """
    # Guard: skip if labels already exist
    existing = list(Path(labels_dir).rglob('*.txt'))
    if existing:
        print(f"Labels already exist — {len(existing):,} .txt files found. Skipping.")
        return

    bbox_files = sorted(Path(soccernet_dir).rglob('*player_boundingbox_maskrcnn.json'))
    print(f"Found {len(bbox_files)} MaskRCNN annotation files.")

    total_labels = 0
    for i, bbox_file in enumerate(bbox_files):
        match_name    = bbox_file.parent.name.replace(' ', '_').replace('/', '_')
        half          = bbox_file.stem.split('_')[0]  # "1" or "2"
        labels_folder = Path(labels_dir) / f"{match_name}_{half}_224p"
        print(f"[{i+1}/{len(bbox_files)}] {bbox_file.parent.name} — Half {half}")
        count         = convert_bbox_file(str(bbox_file), str(labels_folder))
        total_labels += count
        print(f"  {count} label files created")

    print(f"\nTotal YOLO label files created: {total_labels:,}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("Data Preprocessing Script")
    print("=" * 50)
    print("Step 1: Check frame size")
    check_frame_size(FRAMES_DIR)
    print("\nStep 2: Extract frames (skips if already done)")
    extract_all_frames(SOCCERNET, FRAMES_DIR, fps_extract=2)
    print("\nStep 3: Convert MaskRCNN boxes to YOLO format (skips if already done)")
    convert_all_bbox_files(SOCCERNET, LABELS_DIR)
