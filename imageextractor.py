#!/usr/bin/env python3
import cv2, os
import numpy as np

# --------- SETTINGS ---------
inp = "/Users/vid/Desktop/gesture/raw-vid/2new-set_5.MOV"     # video input
outdir = "frames/fist"
os.makedirs(outdir, exist_ok=True)

# Threshold for sharpness (tune this)
SHARPNESS_THRESHOLD = 20.0

# ----------------------------

# Use FFMPEG backend for better .mov support
cap = cv2.VideoCapture(inp, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

idx = 0          # frame index (all frames)
saved = 0        # number of sharp frames saved

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Convert to gray for sharpness check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute variance of Laplacian (focus measure)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Keep only frames with enough sharpness
    if fm >= SHARPNESS_THRESHOLD:

        # ---- SQUARE CROP ----
        h, w = frame.shape[:2]
        side = min(h, w)

        # center crop
        top = (h - side) // 2
        left = (w - side) // 2
        square = frame[top:top+side, left:left+side]

        # Save cropped square frame
        cv2.imwrite(os.path.join(outdir, f"frame_{idx:06d}.jpg"), square)
        saved += 1

    idx += 1

cap.release()
print(f"Total frames: {idx}, saved sharp frames: {saved}")
