# Industrial AI Quality Control System

A real-time industrial quality control vision system built with Python, OpenCV, and YOLOv8. Detects, classifies, and inspects products automatically — making instant Pass/Fail decisions with a professional HMI dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## What It Does

- Detects real objects in real time using YOLOv8
- Makes instant Pass/Fail quality decisions
- Displays a live professional HMI dashboard
- Automatically photographs and logs every detection
- Tracks production statistics live
- Generates a full session report on completion

---

## Demo

### PASS Detection
Object detected, size verified, confidence confirmed — product passes quality check.

### FAIL Detection
Low confidence or size defect detected — product flagged and logged automatically.

---

## System Architecture

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| OpenCV 4.x | Camera feed, image processing, display |
| YOLOv8 (Ultralytics) | Real time object detection |
| NumPy | Image array manipulation |

---

## Project Files

| File | Description |
|---|---|
| `industrial_qc_system.py` | Final complete system |
| `hmi_dashboard.py` | HMI dashboard overlay |
| `defect_logger.py` | Auto defect image logging |
| `quality_control.py` | Pass/Fail counter system |
| `defect_detection.py` | Color and shape defect detection |
| `yolo_tracking.py` | Real time object tracking |
| `people_counter.py` | Zone monitoring system |
| `yolo_filter.py` | Filtered object detection |
| `yolo_test.py` | Basic YOLO detection |
| `vision_system.py` | Combined vision pipeline |
| `webcam.py` | Live webcam processing |
| `contours.py` | Shape recognition |
| `color_detection.py` | Color isolation |
| `edges.py` | Edge detection |
| `test.py` | Environment test |

---

## Installation

**1 — Clone the repository**
```bash
git clone https://github.com/Dammbobo/computer-vision-journey.git
cd computer-vision-journey
```

**2 — Install dependencies**
```bash
pip install opencv-python
pip install numpy
pip install ultralytics
pip install matplotlib
```

**3 — Run the system**
```bash
python industrial_qc_system.py
```

---

## How It Works

### Object Detection
YOLOv8 processes every camera frame in real time, identifying objects from a configurable target list. Each detection includes an object label and confidence score.

### Quality Analysis
Each detected object is evaluated against quality standards:
- **Size check** — is the object within acceptable size range?
- **Confidence check** — is YOLO confident enough about what it sees?
- **Shape check** — does the object meet shape requirements?

### Pass/Fail Decision
If all checks pass → **PASS** (green)
If any check fails → **FAIL** (red) with specific defect logged

### Auto Logging
Every status change triggers an automatic photo save:
- PASS images → `qc_system_logs/passed/`
- FAIL images → `qc_system_logs/defects/`
- Filenames include object name, defect type and timestamp

---

## Industrial Applications

This system demonstrates the same technology used in:

- **Automotive manufacturing** — part inspection on assembly lines
- **Pharmaceutical production** — pill and packaging verification
- **Food processing** — product quality and contamination detection
- **Electronics manufacturing** — PCB and component inspection
- **Packaging lines** — label and seal verification

---

## Background

Built as part of a 15-day Computer Vision learning journey by **Oluwadamilola Adekanye** — a Robotics Technician and Automation Programmer with hands-on experience programming industrial robots (ABB, Fanuc, Kawasaki, Nachi, Motoman) at PWO Canada and Toyota Motor Manufacturing Canada.

This project bridges the gap between physical industrial automation and AI vision systems — combining factory floor knowledge with modern computer vision technology.

---

## Connect

- LinkedIn: [linkedin.com/in/damiadekanye](https://linkedin.com/in/damiadekanye)
- GitHub: [github.com/Dammbobo](https://github.com/Dammbobo)

---

## License

MIT License — feel free to use and build on this project.