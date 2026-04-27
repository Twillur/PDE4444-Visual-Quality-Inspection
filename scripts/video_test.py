"""
video_test.py
-------------
Runs the trained tissue-box inspection model on a video file.
Produces an annotated output video and a per-frame prediction log.

Usage (run from project root):
    py -3.11 scripts/video_test.py videos/IMG_0931.MOV
    py -3.11 scripts/video_test.py videos/IMG_0932.MOV

Output:
    models/demo_captures/<original_name>_result.mp4   — annotated video
    Console: frame-by-frame predictions + final verdict
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH           = "models/best_model.pth"
CLASSES              = ["fail", "pass"]          # must match ImageFolder sort order
IMG_SIZE             = 224
CONFIDENCE_THRESHOLD = 0.75
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
FRAME_SKIP           = 5   # process every Nth frame (speeds up long videos)

# ── Colours (BGR for OpenCV) ──────────────────────────────────────────────────
GREEN = (50, 205, 50)
RED   = (50, 50, 220)
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
GRAY  = (180, 180, 180)

# ── Transform ─────────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("[video_test] Model loaded.")
    return model

# ── Predict single frame ──────────────────────────────────────────────────────
def predict(model, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)
    tensor    = infer_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
    return CLASSES[pred_idx.item()], conf.item()

# ── Draw overlay ──────────────────────────────────────────────────────────────
def draw_overlay(frame, label, confidence, frame_no, total_frames):
    h, w = frame.shape[:2]

    if confidence < CONFIDENCE_THRESHOLD:
        text  = f"Analysing...  {confidence*100:.0f}%"
        color = GRAY
    elif label == "pass":
        text  = f"PASS  {confidence*100:.0f}%"
        color = GREEN
    else:
        text  = f"FAIL  {confidence*100:.0f}%"
        color = RED

    # Inspection box in centre
    box_size = int(min(h, w) * 0.6)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 65), BLACK, -1)
    cv2.putText(frame, text, (20, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

    # Frame counter (bottom right)
    counter_text = f"Frame {frame_no}/{total_frames}"
    cv2.rectangle(frame, (0, h - 35), (w, h), BLACK, -1)
    cv2.putText(frame, counter_text, (w - 200, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Tissue Box Inspection — PDE4444", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    return frame

# ── Main ──────────────────────────────────────────────────────────────────────
def main(video_path):
    if not os.path.isfile(video_path):
        print(f"[video_test] ERROR: File not found: {video_path}")
        sys.exit(1)

    model = load_model()
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[video_test] ERROR: Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[video_test] Video: {video_path}")
    print(f"             {width}x{height} @ {fps:.1f} fps  |  {total_frames} frames")

    # Output file
    save_dir   = "models/demo_captures"
    os.makedirs(save_dir, exist_ok=True)
    base_name  = os.path.splitext(os.path.basename(video_path))[0]
    out_path   = os.path.join(save_dir, f"{base_name}_result.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    predictions = []   # (frame_no, label, confidence)
    frame_no    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        if frame_no % FRAME_SKIP == 0 or frame_no == 1:
            label, confidence = predict(model, frame)
            predictions.append((frame_no, label, confidence))
            if frame_no % 30 == 0:
                print(f"  Frame {frame_no:4d}/{total_frames}  →  {label.upper():4s}  ({confidence*100:.1f}%)")
        else:
            # Reuse last prediction for skipped frames
            if predictions:
                _, label, confidence = predictions[-1]
            else:
                label, confidence = "pass", 0.0

        annotated = draw_overlay(frame.copy(), label, confidence, frame_no, total_frames)
        writer.write(annotated)

    cap.release()
    writer.release()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  VIDEO: {os.path.basename(video_path)}")
    print("="*55)

    if not predictions:
        print("  No frames were processed.")
        return

    labels_only = [p[1] for p in predictions]
    confs_only  = [p[2] for p in predictions]
    counts      = Counter(labels_only)
    avg_conf    = sum(confs_only) / len(confs_only)
    fail_pct    = counts["fail"] / len(labels_only) * 100
    pass_pct    = counts["pass"] / len(labels_only) * 100

    print(f"  Frames analysed : {len(predictions)}")
    print(f"  PASS frames     : {counts['pass']}  ({pass_pct:.1f}%)")
    print(f"  FAIL frames     : {counts['fail']}  ({fail_pct:.1f}%)")
    print(f"  Avg confidence  : {avg_conf*100:.1f}%")
    print()

    # Verdict: FAIL if >30% of frames flagged as damaged
    FAIL_THRESHOLD = 30.0
    if fail_pct > FAIL_THRESHOLD:
        verdict       = "FAIL — DAMAGED"
        verdict_color = "RED"
    else:
        verdict       = "PASS — GOOD CONDITION"
        verdict_color = "GREEN"

    print(f"  VERDICT  :  {verdict}")
    print(f"  Output   :  {out_path}")
    print("="*55 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.11 scripts/video_test.py <path_to_video>")
        print("Example: py -3.11 scripts/video_test.py videos/IMG_0931.MOV")
        sys.exit(1)
    main(sys.argv[1])
