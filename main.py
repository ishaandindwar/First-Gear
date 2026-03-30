"""
FINAL VERSION: Learned Motion + Multi-Frame + Probabilities

Pipeline:
Multiple Images → YOLO Detection → Tracking → Learn Motion (Linear Model) → Predict Futures → Probabilities → ADE/FDE

Runs locally on any computer.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
model_yolo = YOLO("yolov8n.pt")

# =========================
# DETECT CENTERS
# =========================
def detect_centers(folder):
    files = sorted(os.listdir(folder))[:10]
    all_centers = []
    last_frame = None

    for f in files:
        path = os.path.join(folder, f)
        frame = cv2.imread(path)
        last_frame = frame

        results = model_yolo(frame)
        centers = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 0:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    centers.append([cx, cy])

        all_centers.append(centers)

    return all_centers, last_frame

# =========================
# SIMPLE TRACKING
# =========================
def build_track(all_centers):
    track = []
    for centers in all_centers:
        if len(centers) > 0:
            track.append(centers[0])
    return np.array(track)

# =========================
# LEARN MOTION (Linear Fit)
# =========================
def learn_motion(track):
    t = np.arange(len(track))

    x = track[:,0]
    y = track[:,1]

    # Fit linear model
    vx = np.polyfit(t, x, 1)[0]
    vy = np.polyfit(t, y, 1)[0]

    return vx, vy

# =========================
# GENERATE FUTURE (LEARNED + VARIATION)
# =========================
def generate_futures(track, vx, vy, steps=30, num_paths=3):
    futures = []
    last_x, last_y = track[-1]

    for k in range(num_paths):
        traj = []
        x, y = last_x, last_y

        for t in range(steps):
            # controlled variation
            noise_scale = 0.2 * (k+1)
            dx = vx + np.random.randn() * noise_scale
            dy = vy + np.random.randn() * noise_scale

            x += dx
            y += dy
            traj.append([x, y])

        futures.append(np.array(traj))

    return futures

# =========================
# FAKE GROUND TRUTH
# =========================
def create_ground_truth(track, vx, vy, steps=30):
    x, y = track[-1]
    gt = []

    for i in range(steps):
        x += vx
        y += vy
        gt.append([x, y])

    return np.array(gt)

# =========================
# METRICS
# =========================
def compute_ade(pred, gt):
    return np.mean(np.linalg.norm(pred - gt, axis=1))


def compute_fde(pred, gt):
    return np.linalg.norm(pred[-1] - gt[-1])

# =========================
# PROBABILITIES
# =========================
def compute_probabilities(futures, gt):
    ades = np.array([compute_ade(f, gt) for f in futures])

    scores = 1 / (ades + 1e-6)
    probs = scores / scores.sum()

    return probs, ades

# =========================
# VISUALIZE
# =========================
def visualize(frame, track, futures, gt, probs):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # past
    plt.plot(track[:,0], track[:,1], 'b', label='Past')

    # ground truth
    plt.plot(gt[:,0], gt[:,1], 'g', label='Ground Truth')

    # predictions
    for i, f in enumerate(futures):
        plt.plot(f[:,0], f[:,1], '--', label=f'Pred {i} ({probs[i]*100:.1f}%)')

    plt.legend()
    plt.title("Learned Trajectory Prediction")
    plt.show()

# =========================
# MAIN
# =========================
def run(folder):
    all_centers, frame = detect_centers(folder)

    track = build_track(all_centers)

    if len(track) < 5:
        print("Not enough data")
        return

    vx, vy = learn_motion(track)

    futures = generate_futures(track, vx, vy)
    gt = create_ground_truth(track, vx, vy)

    probs, ades = compute_probabilities(futures, gt)
    fdes = [compute_fde(f, gt) for f in futures]

    print("\n--- RESULTS ---")
    for i in range(len(futures)):
        print(f"Path {i}: ADE={ades[i]:.2f}, FDE={fdes[i]:.2f}, Prob={probs[i]*100:.2f}%")

    print("Best ADE:", min(ades))
    print("Best FDE:", min(fdes))

    visualize(frame, track, futures, gt, probs)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run("frames")
