"""
demo.py
-------
Live camera demo. Captures frames from a webcam (or phone via DroidCam),
runs the trained model, and overlays PASS / FAIL on the screen in real time.

Run from project root:
    py -3.11 scripts/demo.py

Controls:
    Q  →  quit
    S  →  save current frame + prediction to models/demo_captures/
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/best_model.pth"
CLASSES      = ["fail", "pass"]          # must match ImageFolder sort order
IMG_SIZE     = 224
CAMERA_INDEX = 0                         # 0 = default webcam; try 1 for DroidCam
CONFIDENCE_THRESHOLD = 0.75              # only show label if model is confident
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Colours (BGR for OpenCV) ──────────────────────────────────────────────────
GREEN = (50, 205, 50)
RED   = (50, 50, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# ── Transform for inference ───────────────────────────────────────────────────
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
    print("[demo] Model loaded successfully.")
    return model

# ── Predict single frame ──────────────────────────────────────────────────────
def predict(model, frame_bgr):
    """Convert OpenCV BGR frame → PIL → tensor → model → label + confidence."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)
    tensor    = infer_transform(pil_img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

    label      = CLASSES[pred_idx.item()]
    confidence = conf.item()
    return label, confidence

# ── Draw overlay on frame ─────────────────────────────────────────────────────
def draw_overlay(frame, label, confidence):
    h, w = frame.shape[:2]

    if confidence < CONFIDENCE_THRESHOLD:
        text  = "Analysing..."
        color = WHITE
    elif label == "pass":
        text  = f"PASS  {confidence*100:.0f}%"
        color = GREEN
    else:
        text  = f"FAIL  {confidence*100:.0f}%"
        color = RED

    # Draw inspection box in centre
    box_size = int(min(h, w) * 0.6)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Draw label banner at top
    cv2.rectangle(frame, (0, 0), (w, 60), BLACK, -1)
    cv2.putText(frame, text, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

    # Draw instructions at bottom
    cv2.rectangle(frame, (0, h - 35), (w, h), BLACK, -1)
    cv2.putText(frame, "Q: quit    S: save frame", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model()
    cap   = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[demo] ERROR: Could not open camera index {CAMERA_INDEX}.")
        print("       Try changing CAMERA_INDEX to 1 or 2 at the top of the script.")
        exit(1)

    print("[demo] Camera opened. Press Q to quit, S to save a frame.")
    save_dir = "models/demo_captures"
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[demo] Failed to grab frame.")
            break

        label, confidence = predict(model, frame)
        frame = draw_overlay(frame, label, confidence)

        cv2.imshow("Tissue Box Inspection — PDE4444", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"{label}_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[demo] Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[demo] Done.")
