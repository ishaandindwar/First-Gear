# 🚶 Multi-Modal Trajectory Prediction (Baseline)

A real-time system for predicting **multiple possible future trajectories of pedestrians** using computer vision and probabilistic motion modeling.

---

## 📌 Overview

In dynamic urban environments, predicting a single future path is not sufficient due to the uncertainty in human behavior.

This project implements a **multi-modal trajectory prediction system** that:

* Extracts motion from video frames
* Learns movement trends
* Generates **multiple possible future paths**
* Assigns **probabilities** to each prediction

---

## 🎯 Features

* 🧠 Multi-modal trajectory prediction (3 paths)
* 🎥 Pedestrian detection using YOLOv8
* 🔁 Multi-frame tracking
* 📈 Motion learning using linear regression
* 🎲 Probabilistic future generation
* 📊 ADE & FDE evaluation
* ⚡ Runs locally (no training required)

---

## 🧠 Methodology

### 1. Detection

* Detect pedestrians in each frame using YOLO
* Extract bounding box centers `(x, y)`

### 2. Tracking

* Track a single pedestrian across frames
* Uses nearest-neighbor matching

### 3. Motion Learning

* Estimate velocity using linear regression:

```
vx, vy = rate of change of position
```

### 4. Multi-Modal Prediction

* Generate multiple trajectories using controlled randomness:

```
dx = vx + noise
dy = vy + noise
```

* Different noise levels produce different behaviors

### 5. Probability Estimation

* Based on trajectory error (ADE):

```
Score = 1 / ADE
```

---

## 🔄 Pipeline

```
Frames → Detection → Tracking → Motion Learning → Multi-Path Prediction → Probabilities → Evaluation
```

---

## 📊 Example Output

Each run generates:

* 3 predicted trajectories
* Probability for each path
* ADE & FDE scores

Example:

```
Path 0: ADE=12.34, FDE=20.11, Prob=52.3%
Path 1: ADE=15.67, FDE=25.44, Prob=30.1%
Path 2: ADE=22.89, FDE=35.67, Prob=17.6%
```

---

## 📁 Project Structure

```
trajectory-prediction/
│
├── frames/          # Input image sequence
├── outputs/         # Saved results
├── main.py          # Main script
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

```bash
pip install opencv-python numpy matplotlib torch ultralytics
```

---

## ▶️ Usage

1. Add frames to `frames/` folder:

```
frames/
 ├── 1.jpg
 ├── 2.jpg
 ├── 3.jpg
```

2. Run:

```bash
python main.py
```

---

## 📊 Evaluation Metrics

* **ADE (Average Displacement Error)**
* **FDE (Final Displacement Error)**

Lower values indicate better predictions.

---

## ⚠️ Limitations

* Tracks only one pedestrian
* Assumes linear motion
* No social interaction modeling
* Uses approximated ground truth

---

## 🔮 Future Work

* LSTM / Transformer-based models
* Multi-agent interaction modeling
* Training on real datasets (e.g., nuScenes)
* Diffusion-based trajectory generation

---

## 🏆 Hackathon Pitch

> We built a real-time multi-modal trajectory prediction system that models pedestrian motion using learned velocity patterns and probabilistic sampling, enabling multiple plausible future predictions for safer decision-making.

---

## 📜 License

MIT License
