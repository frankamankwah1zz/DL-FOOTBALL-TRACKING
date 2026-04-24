"""
simple_tracker.py
=================
IoU-based multi-object tracker and tracking pipeline.

Extracted from:
  Cell 58 — ByteTrack tracking using live YOLOv8 detections
             (model.track with bytetrack.yaml, 3× upscaling of 224p frames)
  Cell 61 — SimpleTracker on SoccerNet MaskRCNN pre-computed bounding boxes
             (produces the 9.4 avg detections/frame result in the paper)
  Cell 79 — Tracking video generator (produces tracking_demo.mp4)

Two tracking approaches used in this project
--------------------------------------------
Approach A (Cell 58):
    - Run YOLOv8 model.track() with Ultralytics' built-in ByteTrack
    - 3× upscale each 224p frame before detection
    - Saves per-match tracking JSON and annotated .mp4

Approach B (Cell 61 — produces the paper result):
    - Use SoccerNet's pre-computed MaskRCNN bounding boxes (no live detection)
    - Run SimpleTracker (IoU-based) to assign consistent IDs
    - Result: 9.4 avg detections/frame across 3 EPL matches (1,500 frames)

Approach C (Cell 79):
    - Demo video: YOLO model.predict() + SimpleTracker → tracking_demo.mp4

Classes (matching the detection model):
    0 = ball
    1 = goalkeeper
    2 = player
    3 = referee

Dependencies: ultralytics, opencv-python, numpy
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


# ── Class mapping ──────────────────────────────────────────────
CLASS_NAMES  = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
CLASS_COLORS = {
    'ball'      : (0,   0,   255),   # red
    'goalkeeper': (255, 165,   0),   # orange
    'player'    : (0,   255,   0),   # green
    'referee'   : (255,   0,   0),   # blue
}


# ============================================================
# SimpleTracker — IoU-based tracker (Cell 61 and Cell 79)
# ============================================================

class SimpleTracker:
    """
    Lightweight IoU-based multi-object tracker.

    Assigns consistent integer track IDs across video frames by matching
    new detections to existing tracks using Intersection over Union (IoU).
    Tracks that are unmatched for more than `max_missing` frames are removed.

    Two variants used in the notebook:

    Cell 61 (MaskRCNN boxes):
        detections = list of [x1, y1, x2, y2]
        update() returns list of (track_id, bbox)

    Cell 79 (YOLO demo video):
        detections = list of [x1, y1, x2, y2, class_id, conf]
        update() returns dict {track_id: detection}
    """

    def __init__(self, iou_threshold=0.3, max_missing=5, iou_thresh=None):
        """
        Args:
            iou_threshold (float): Min IoU to match detection to track. Default 0.3.
            max_missing   (int):   Frames a track can go unmatched before deletion. Default 5.
            iou_thresh    (float): Alias used in Cell 79 (iou_thresh=0.35). If provided,
                                   overrides iou_threshold.
        """
        self.iou_threshold = iou_thresh if iou_thresh is not None else iou_threshold
        self.max_missing   = max_missing
        self.next_id       = 1
        self.tracks        = {}              # track_id → last detection/bbox
        self.missing       = defaultdict(int)  # track_id → consecutive missed frames

    def iou(self, boxA, boxB):
        """
        Compute Intersection over Union between two [x1, y1, x2, y2] boxes.

        Args:
            boxA, boxB: Sequences of (x1, y1, x2, y2) pixel coordinates.

        Returns:
            float: IoU in [0.0, 1.0].
        """
        ax1, ay1, ax2, ay2 = boxA[0], boxA[1], boxA[2], boxA[3]
        bx1, by1, bx2, by2 = boxB[0], boxB[1], boxB[2], boxB[3]

        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter)

    def update(self, detections):
        """
        Match detections to tracks and return assignments.

        Supports two calling conventions (auto-detected):
          Plain boxes  [x1,y1,x2,y2]              → returns list of (tid, bbox)
          With class   [x1,y1,x2,y2,class_id,conf] → returns dict {tid: detection}

        Args:
            detections (list): Per-frame detection list.

        Returns:
            list | dict: Track assignments for this frame.
        """
        if not detections:
            for tid in list(self.tracks.keys()):
                self.missing[tid] += 1
                if self.missing[tid] > self.max_missing:
                    del self.tracks[tid]
                    del self.missing[tid]
            return [] if (not detections or len(detections[0]) <= 4) else {}

        use_plain = len(detections[0]) <= 4
        return self._update_plain(detections) if use_plain else self._update_with_class(detections)

    # ── Cell 61: plain [x1,y1,x2,y2] boxes ──────────────────────

    def _update_plain(self, detections):
        """Returns list of (track_id, bbox)."""
        if not self.tracks:
            results = []
            for det in detections:
                tid = self.next_id
                self.tracks[tid]  = det
                self.missing[tid] = 0
                self.next_id += 1
                results.append((tid, det))
            return results

        track_ids       = list(self.tracks.keys())
        track_boxes     = [self.tracks[t] for t in track_ids]
        assigned_tracks = set()
        results         = []

        for di, det in enumerate(detections):
            best_score = self.iou_threshold
            best_tid   = None
            best_ti    = None

            for ti, (tid, tbox) in enumerate(zip(track_ids, track_boxes)):
                if ti in assigned_tracks:
                    continue
                score = self.iou(det, tbox)
                if score > best_score:
                    best_score = score
                    best_tid   = tid
                    best_ti    = ti

            if best_tid is not None:
                self.tracks[best_tid]  = det
                self.missing[best_tid] = 0
                assigned_tracks.add(best_ti)
                results.append((best_tid, det))
            else:
                tid = self.next_id
                self.tracks[tid]  = det
                self.missing[tid] = 0
                self.next_id += 1
                results.append((tid, det))

        matched_tids = {r[0] for r in results}
        for tid in list(self.tracks.keys()):
            if tid not in matched_tids:
                self.missing[tid] += 1
                if self.missing[tid] > self.max_missing:
                    del self.tracks[tid]
                    del self.missing[tid]

        return results

    # ── Cell 79: [x1,y1,x2,y2,class_id,conf] boxes ──────────────

    def _update_with_class(self, detections):
        """Returns dict {track_id: detection}."""
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = det
                self.next_id += 1
            return {i + 1: d for i, d in enumerate(detections)}

        track_ids   = list(self.tracks.keys())
        track_boxes = [self.tracks[t][:4] for t in track_ids]
        assigned    = {}
        used_dets   = set()

        for t_idx, tid in enumerate(track_ids):
            best_score = self.iou_threshold
            best_det   = -1
            for d_idx, det in enumerate(detections):
                if d_idx in used_dets:
                    continue
                score = self.iou(track_boxes[t_idx], det[:4])
                if score > best_score:
                    best_score = score
                    best_det   = d_idx
            if best_det >= 0:
                assigned[tid]         = detections[best_det]
                self.tracks[tid]      = detections[best_det]
                self.missing[tid]     = 0
                used_dets.add(best_det)
            else:
                self.missing[tid] += 1
                if self.missing[tid] > self.max_missing:
                    del self.tracks[tid]

        for d_idx, det in enumerate(detections):
            if d_idx not in used_dets:
                self.tracks[self.next_id]  = det
                assigned[self.next_id]     = det
                self.next_id += 1

        return assigned


# ============================================================
# Frame upscaling utility (Cell 58)
# ============================================================

def upscale_frame(frame_path, scale=3):
    """
    Upscale a low-resolution SoccerNet frame (398×224 px) before detection.

    SoccerNet videos are 224p. At native resolution, player bounding boxes
    are only ~10–20 px tall, too small for reliable YOLOv8 detection.
    Upscaling 3× to ~1194×672 significantly improves recall.

    Args:
        frame_path (str): Path to the .jpg frame file.
        scale      (int): Upscale factor. Default 3 (398→1194, 224→672).

    Returns:
        np.ndarray | None: Upscaled BGR image, or None if read fails.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


# ============================================================
# MaskRCNN JSON loader (Cell 61)
# ============================================================

def load_maskrcnn_json(json_path):
    """
    Load a SoccerNet MaskRCNN bounding box annotation file.

    File format:
        {
          "predictions": [{"bboxes": [[x1,y1,x2,y2],...], "onfield": [1,0,...]}, ...],
          "size": [num_frames, height, width, channels]
        }

    Args:
        json_path (str): Path to *_player_boundingbox_maskrcnn.json.

    Returns:
        tuple: (predictions list, size list).
    """
    with open(json_path) as f:
        data = json.load(f)
    return data['predictions'], data['size']


def match_name_to_frames(match_name, frames_base):
    """
    Map a SoccerNet match folder name to the extracted frames folder.

    SoccerNet folder : '2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'
    Frames folder    : '2015-02-21_-_18-00_Chelsea_1_-_1_Burnley_1_224p'

    Args:
        match_name  (str): SoccerNet match directory name.
        frames_base (str): Root directory of extracted frame folders.

    Returns:
        Path | None: Matched frames folder, or None if not found.
    """
    normalized = match_name.replace(' ', '_').replace('-', '_')
    for folder in Path(frames_base).iterdir():
        if normalized.lower() in folder.name.replace(' ', '_').replace('-', '_').lower():
            return folder
    return None


# ============================================================
# Approach A: ByteTrack via model.track() (Cell 58)
# ============================================================

def track_with_bytetrack(model_path, frames_dir, output_dir,
                         max_matches=1, max_frames=100,
                         conf=0.10, iou=0.45, upscale=3,
                         save_video=True, save_json=True):
    """
    Run YOLOv8 ByteTrack on extracted SoccerNet frames (Cell 58 logic).

    Uses model.track() with bytetrack.yaml for built-in tracking.
    Upscales each 224p frame by `upscale` factor before inference.

    Args:
        model_path  (str):  Path to fine-tuned .pt model weights.
        frames_dir  (str):  Root directory of extracted frame folders.
        output_dir  (str):  Where to save tracking results.
        max_matches (int):  Number of match folders to process. Default 1.
        max_frames  (int):  Max frames per match. Default 100.
        conf        (float): YOLO confidence threshold. Default 0.10.
        iou         (float): NMS IoU threshold. Default 0.45.
        upscale     (int):  Upscale factor for 224p frames. Default 3.
        save_video  (bool): Save annotated .mp4. Default True.
        save_json   (bool): Save tracking JSON. Default True.
    """
    from ultralytics import YOLO

    TEMP_FRAME = '/content/temp_frame.jpg'
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")

    frames_path   = Path(frames_dir)
    match_folders = sorted(frames_path.iterdir())[:max_matches]
    all_results   = []

    for match_folder in match_folders:
        match_name = match_folder.name
        frames     = sorted(match_folder.glob('*.jpg'))[:max_frames]
        if not frames:
            continue

        print(f"\nProcessing: {match_name}")
        print(f"  Frames   : {len(frames)}")
        print(f"  Upscale  : {upscale}x")

        writer = None
        if save_video:
            sample = upscale_frame(str(frames[0]), upscale)
            if sample is not None:
                h, w   = sample.shape[:2]
                writer = cv2.VideoWriter(
                    os.path.join(output_dir, f"{match_name}_tracked.mp4"),
                    cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h)
                )

        tracking_data = {
            'match': match_name, 'total_frames': len(frames),
            'upscale_factor': upscale, 'frames': []
        }

        for i, frame_path in enumerate(frames):
            upscaled = upscale_frame(str(frame_path), upscale)
            if upscaled is None:
                continue
            cv2.imwrite(TEMP_FRAME, upscaled)

            results = model.track(
                source  = TEMP_FRAME,
                tracker = 'bytetrack.yaml',
                conf    = conf,
                iou     = iou,
                persist = True,
                verbose = False,
                imgsz   = 640,
            )
            result     = results[0]
            frame_img  = upscaled.copy()
            frame_data = {'frame_id': frame_path.stem, 'detections': []}

            if result.boxes is not None and len(result.boxes) > 0:
                boxes     = result.boxes.xyxy.cpu().numpy()
                ids       = result.boxes.id
                classes   = result.boxes.cls.cpu().numpy()
                confs     = result.boxes.conf.cpu().numpy()
                track_ids = ids.cpu().numpy() if ids is not None else [-1] * len(boxes)

                for box, tid, cls, cf in zip(boxes, track_ids, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = CLASS_NAMES.get(int(cls), 'unknown')
                    frame_data['detections'].append({
                        'track_id': int(tid), 'class': cls_name,
                        'bbox': [x1, y1, x2, y2], 'confidence': round(float(cf), 3)
                    })
                    color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                    cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_img, f"{cls_name} #{int(tid)}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            tracking_data['frames'].append(frame_data)
            if writer is not None:
                writer.write(frame_img)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(frames)} frames")

        if writer is not None:
            writer.release()
            print(f"  Video saved")
        if save_json:
            jp = os.path.join(output_dir, f"{match_name}_tracking.json")
            with open(jp, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            print(f"  JSON saved")

        total_d = sum(len(fr['detections']) for fr in tracking_data['frames'])
        avg_d   = total_d / len(frames) if frames else 0
        print(f"  Total detections : {total_d:,}  |  Avg/frame: {avg_d:.1f}")
        all_results.append(tracking_data)

    total_frames = sum(r['total_frames'] for r in all_results)
    total_dets   = sum(sum(len(f['detections']) for f in r['frames']) for r in all_results)
    print(f"\nTracking complete")
    print(f"  Matches : {len(all_results)}  |  Frames: {total_frames:,}"
          f"  |  Avg/frame: {total_dets/total_frames:.1f}" if total_frames else "")


# ============================================================
# Approach B: MaskRCNN + SimpleTracker (Cell 61 — paper result)
# ============================================================

def process_match_with_maskrcnn(maskrcnn_file, frames_base, output_dir,
                                save_video=True, save_json=True, max_frames=500):
    """
    Run SimpleTracker on SoccerNet MaskRCNN pre-computed bounding boxes.

    This is the approach that produced 9.4 avg detections/frame across
    3 EPL matches (1,500 frames) — the tracking result reported in the paper.

    Args:
        maskrcnn_file (str):  Path to *_boundingbox_maskrcnn.json.
        frames_base   (str):  Root directory of extracted frame folders.
        output_dir    (str):  Where to save output files.
        save_video    (bool): Save annotated .mp4. Default True.
        save_json     (bool): Save tracking JSON. Default True.
        max_frames    (int):  Max frames to process. Default 500.

    Returns:
        dict | None: Tracking results, or None if match folder not found.
    """
    match_name = Path(maskrcnn_file).parent.name
    print(f"\nProcessing: {match_name}")

    predictions, size = load_maskrcnn_json(maskrcnn_file)
    print(f"  Annotated frames: {len(predictions):,}")
    print(f"  Original res    : {size[2]}×{size[1]}")

    frames_folder = match_name_to_frames(match_name, frames_base)
    if frames_folder is None:
        print(f"  No matching frames folder found — skipping")
        return None
    print(f"  Frames folder: {frames_folder.name}")

    frames     = sorted(frames_folder.glob('*.jpg'))
    use_frames = min(max_frames, len(frames), len(predictions))
    frames     = frames[:use_frames]
    print(f"  Processing {use_frames} frames")

    # Scale from MaskRCNN resolution (1920×1080) to frame resolution (398×224)
    scale_x = 398 / size[2]
    scale_y = 224 / size[1]

    tracker = SimpleTracker(iou_threshold=0.3, max_missing=5)

    writer = None
    if save_video:
        sample = cv2.imread(str(frames[0]))
        if sample is not None:
            h, w   = sample.shape[:2]
            writer = cv2.VideoWriter(
                os.path.join(output_dir, f"{frames_folder.name}_tracked.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h)
            )

    tracking_data = {
        'match': match_name, 'total_frames': use_frames,
        'frames_processed': 0, 'frames': []
    }

    for i, frame_path in enumerate(frames):
        pred      = predictions[i]
        bboxes    = pred['bboxes']
        onfield   = pred.get('onfield', [1] * len(bboxes))
        onfield_b = [b for b, on in zip(bboxes, onfield) if on == 1]

        # Scale boxes to frame resolution
        scaled = [
            [int(b[0]*scale_x), int(b[1]*scale_y),
             int(b[2]*scale_x), int(b[3]*scale_y)]
            for b in onfield_b
        ]

        tracked    = tracker.update(scaled)
        frame_img  = cv2.imread(str(frame_path))
        frame_data = {'frame_id': frame_path.stem, 'detections': []}

        for tid, bbox in tracked:
            x1, y1, x2, y2 = bbox
            frame_data['detections'].append(
                {'track_id': tid, 'class': 'player', 'bbox': [x1, y1, x2, y2]}
            )
            if frame_img is not None:
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame_img, f"#{tid}", (x1, y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        tracking_data['frames'].append(frame_data)
        tracking_data['frames_processed'] += 1
        if writer is not None and frame_img is not None:
            writer.write(frame_img)
        if (i + 1) % 100 == 0:
            print(f"  Frame {i+1}/{use_frames} | dets: {len(frame_data['detections'])}")

    if writer is not None:
        writer.release()
        print("  Video saved")
    if save_json:
        jp = os.path.join(output_dir, f"{frames_folder.name}_tracking.json")
        with open(jp, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print("  JSON saved")

    total_d = sum(len(f['detections']) for f in tracking_data['frames'])
    avg_d   = total_d / use_frames if use_frames else 0
    print(f"  Total: {total_d:,}  |  Avg/frame: {avg_d:.1f}")
    return tracking_data


def track_all_matches_maskrcnn(soccernet_dir, frames_dir, output_dir,
                               max_matches=3, max_frames=500):
    """
    Run MaskRCNN + SimpleTracker on multiple matches (Cell 61 main block).

    Args:
        soccernet_dir (str): Root SoccerNet directory.
        frames_dir    (str): Root extracted frames directory.
        output_dir    (str): Output directory for results.
        max_matches   (int): Number of matches to process. Default 3.
        max_frames    (int): Max frames per match. Default 500.
    """
    os.makedirs(output_dir, exist_ok=True)
    maskrcnn_files = sorted(
        Path(soccernet_dir).rglob('*boundingbox_maskrcnn.json')
    )[:max_matches]

    print(f"Found {len(maskrcnn_files)} MaskRCNN files")
    print(f"Processing {max_matches} matches, {max_frames} frames each\n")

    all_results = []
    for mf in maskrcnn_files:
        result = process_match_with_maskrcnn(
            str(mf), frames_dir, output_dir, max_frames=max_frames
        )
        if result:
            all_results.append(result)

    total_frames = sum(r['total_frames'] for r in all_results)
    total_dets   = sum(sum(len(f['detections']) for f in r['frames'])
                       for r in all_results)
    print(f"\n{'='*50}")
    print(f"TRACKING COMPLETE")
    print(f"  Matches  : {len(all_results)}")
    print(f"  Frames   : {total_frames:,}")
    print(f"  Total det: {total_dets:,}")
    if total_frames > 0:
        print(f"  Avg/frame: {total_dets/total_frames:.1f}")


# ============================================================
# Approach C: Demo video generator (Cell 79)
# ============================================================

def generate_tracking_video(model_path, frames_dir, output_video,
                            match_folder=None, max_frames=300,
                            conf=0.15, iou=0.45, fps=30,
                            output_size=(960, 540)):
    """
    Run YOLO detect + SimpleTracker and save an annotated demo .mp4.

    This is the Cell 79 logic, packaged as a reusable function.
    The output tracking_demo.mp4 was used in the final submission.

    Args:
        model_path   (str): Path to fine-tuned .pt weights.
        frames_dir   (str): Root directory of extracted frame folders.
        output_video (str): Output .mp4 file path.
        match_folder (str|None): Specific match subfolder, or None for first.
        max_frames   (int): Max frames to process. Default 300.
        conf         (float): YOLO confidence threshold. Default 0.15.
        iou          (float): NMS IoU threshold. Default 0.45.
        fps          (int):  Output video FPS. Default 30.
        output_size  (tuple): Output (width, height). Default (960, 540).
    """
    from ultralytics import YOLO

    print("Loading model...")
    model   = YOLO(model_path)
    tracker = SimpleTracker(iou_threshold=0.35)
    print("Model loaded")

    frames_root = Path(frames_dir)
    match_dir   = (frames_root / match_folder
                   if match_folder else sorted(frames_root.iterdir())[0])
    frame_paths = sorted(match_dir.glob('*.jpg'))[:max_frames]
    print(f"Processing {len(frame_paths)} frames from: {match_dir.name}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, output_size)

    # Per-ID colour — seeded so same ID always gets same colour
    id_colors = {}
    def get_id_color(tid):
        if tid not in id_colors:
            np.random.seed(tid * 37)
            id_colors[tid] = tuple(int(c) for c in np.random.randint(100, 255, 3))
        return id_colors[tid]

    for i, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)

        results = model.predict(
            source=img, conf=conf, iou=iou, imgsz=640, verbose=False
        )[0]

        # Build detection list [x1, y1, x2, y2, class_id, conf]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls  = int(box.cls[0].cpu().numpy())
                cf   = float(box.conf[0].cpu().numpy())
                detections.append([x1, y1, x2, y2, cls, cf])

        tracked = tracker.update(detections)

        for tid, det in tracked.items():
            x1, y1, x2, y2, cls, cf = det
            cls   = int(cls)
            color = get_id_color(tid)
            label = f"ID{tid} {CLASS_NAMES.get(cls, 'obj')} {cf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(img, f"Frame {i+1}/{len(frame_paths)}  |  Tracked: {len(tracked)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(img)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(frame_paths)} frames...")

    writer.release()
    print(f"\nTracking video saved: {output_video}")


# ============================================================
# Entry point — smoke test
# ============================================================

if __name__ == "__main__":
    print("SimpleTracker — smoke test")
    print("=" * 40)
    tracker = SimpleTracker(iou_threshold=0.3)
    f1 = [[10, 10, 50, 50], [100, 100, 150, 150]]
    f2 = [[12, 12, 52, 52], [102, 102, 152, 152]]
    f3 = [[14, 14, 54, 54]]
    for idx, dets in enumerate([f1, f2, f3], 1):
        r = tracker.update(dets)
        print(f"Frame {idx}: {r}")
    print("Expected: IDs persist frame 1→2, one ID drops in frame 3")
