import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# INDUSTRIAL AI QUALITY CONTROL SYSTEM
# Built with OpenCV, YOLO and Python
# Author: Oluwadamilola Adekanye
# ============================================================

# Load YOLO model
model = YOLO("yolov8n.pt")

# Camera
cap = cv2.VideoCapture(0)

# Quality standards
MIN_AREA = 3000
MAX_AREA = 100000
TARGET_OBJECTS = ["cup", "bottle", "cell phone", 
                  "keyboard", "mouse", "book"]

# Production counters
total_inspected = 0
total_passed = 0
total_failed = 0
last_status = None
session_start = datetime.now()

# Save directories
SAVE_DIR = "qc_system_logs"
DEFECT_DIR = os.path.join(SAVE_DIR, "defects")
PASS_DIR = os.path.join(SAVE_DIR, "passed")
os.makedirs(DEFECT_DIR, exist_ok=True)
os.makedirs(PASS_DIR, exist_ok=True)

print("=" * 50)
print("INDUSTRIAL AI QUALITY CONTROL SYSTEM")
print("=" * 50)
print(f"Monitoring: {TARGET_OBJECTS}")
print("Press 'q' to quit")
print("=" * 50)


def draw_hmi_panel(frame, total_inspected, total_passed,
                   total_failed, current_status, 
                   current_object, session_start):

    h, w = frame.shape[:2]
    panel_width = 300
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:] = (15, 15, 15)

    # Header
    cv2.rectangle(panel, (0, 0), (panel_width, 65), (30, 30, 30), -1)
    cv2.putText(panel, "INDUSTRIAL AI", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(panel, "QUALITY CONTROL v2.0", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 200), 1)
    cv2.line(panel, (0, 65), (panel_width, 65), (50, 50, 50), 1)

    # Status box
    status_color = (0, 200, 0) if current_status == "PASS" else \
                   (0, 0, 220) if current_status == "FAIL" else \
                   (80, 80, 80)
    status_text = current_status if current_status else "SCANNING"

    cv2.rectangle(panel, (10, 78), (panel_width-10, 140),
                 status_color, -1)
    cv2.rectangle(panel, (10, 78), (panel_width-10, 140),
                 (255, 255, 255), 1)

    text_x = panel_width//2 - len(status_text) * 11
    cv2.putText(panel, status_text, (text_x, 122),
               cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    # Current object detected
    obj_text = current_object if current_object else "No object"
    cv2.putText(panel, f"Object: {obj_text}", (10, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Divider
    cv2.line(panel, (10, 172), (panel_width-10, 172), (50, 50, 50), 1)

    # Production stats
    cv2.putText(panel, "PRODUCTION STATS", (10, 193),
               cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

    stats = [
        ("Inspected", str(total_inspected), (220, 220, 220)),
        ("Passed", str(total_passed), (0, 220, 0)),
        ("Failed", str(total_failed), (0, 0, 220)),
    ]

    for i, (label, value, color) in enumerate(stats):
        y = 220 + i * 30
        cv2.putText(panel, label, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
        cv2.putText(panel, value, (panel_width-55, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Pass rate
    pass_rate = (total_passed / total_inspected * 100) \
                if total_inspected > 0 else 0
    rate_color = (0, 220, 0) if pass_rate >= 80 else (0, 0, 220)

    cv2.line(panel, (10, 315), (panel_width-10, 315), (50, 50, 50), 1)
    cv2.putText(panel, "Pass Rate", (10, 338),
               cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
    cv2.putText(panel, f"{pass_rate:.1f}%",
               (panel_width-75, 338),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, rate_color, 2)

    # Progress bar
    bar_w = panel_width - 20
    cv2.rectangle(panel, (10, 348), (panel_width-10, 365),
                 (50, 50, 50), -1)
    fill = int(bar_w * pass_rate / 100)
    if fill > 0:
        cv2.rectangle(panel, (10, 348), (10+fill, 365),
                     rate_color, -1)

    # Session info
    cv2.line(panel, (10, 375), (panel_width-10, 375), (50, 50, 50), 1)
    elapsed = datetime.now() - session_start
    mins = int(elapsed.total_seconds() // 60)
    secs = int(elapsed.total_seconds() % 60)

    cv2.putText(panel, "Session", (10, 398),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(panel, f"{mins:02d}:{secs:02d}",
               (panel_width-75, 398),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(panel, "Time", (10, 425),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(panel, current_time, (panel_width-90, 425),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Quality indicator
    cv2.line(panel, (10, 440), (panel_width-10, 440), (50, 50, 50), 1)
    if total_inspected > 0:
        quality_text = "GOOD" if pass_rate >= 80 else "REVIEW"
        quality_color = (0, 220, 0) if pass_rate >= 80 else (0, 100, 220)
        cv2.putText(panel, f"Line Quality: {quality_text}",
                   (10, 463), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, quality_color, 1)

    # Footer
    cv2.rectangle(panel, (0, h-35), (panel_width, h),
                 (30, 30, 30), -1)
    cv2.putText(panel, "AI Vision System v2.0",
               (10, h-12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    combined = np.hstack([frame, panel])
    return combined


while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    current_status = None
    current_object = None
    current_defects = []

    for box in detections:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])

        # Only process target objects
        if label not in TARGET_OBJECTS or confidence < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        area = w * h
        current_object = label

        defects = []

        # Size check
        if area < MIN_AREA:
            defects.append("Too small")
        elif area > MAX_AREA:
            defects.append("Too large")

        # Confidence check
        if confidence < 0.65:
            defects.append(f"Low confidence: {confidence:.2f}")

        # Determine status
        if len(defects) == 0:
            current_status = "PASS"
            color = (0, 220, 0)
        else:
            current_status = "FAIL"
            color = (0, 0, 220)
            current_defects = defects

        # Draw detection box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Label background
        label_bg_y = max(y1-35, 0)
        cv2.rectangle(display,
                     (x1, label_bg_y),
                     (x1 + 200, y1), color, -1)

        # Label text
        cv2.putText(display,
                   f"{label} {confidence:.2f}",
                   (x1+5, y1-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   (255, 255, 255), 1)
        cv2.putText(display, current_status,
                   (x1+5, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   (255, 255, 255), 2)

        # Show defects below box
        for i, defect in enumerate(defects):
            cv2.putText(display, defect,
                       (x1, y2 + 20 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 100, 255), 1)

    # Save on status change
    if current_status is not None and current_status != last_status:
        total_inspected += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if current_status == "PASS":
            total_passed += 1
            filename = f"PASS_{current_object}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(PASS_DIR, filename), display)
            print(f"✅ PASS: {current_object} | "
                  f"Total: {total_inspected}")
        else:
            total_failed += 1
            defect_str = "-".join(current_defects)
            filename = f"FAIL_{current_object}_{defect_str}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(DEFECT_DIR, filename), display)
            print(f"❌ FAIL: {current_object} | "
                  f"Defects: {current_defects} | "
                  f"Total: {total_inspected}")

        last_status = current_status

    # Draw HMI
    output = draw_hmi_panel(
        display, total_inspected, total_passed,
        total_failed, current_status,
        current_object, session_start
    )

    cv2.imshow("Industrial AI Quality Control System", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final session report
pass_rate = (total_passed / total_inspected * 100) \
            if total_inspected > 0 else 0
elapsed = datetime.now() - session_start
duration = f"{int(elapsed.total_seconds()//60):02d}:{int(elapsed.total_seconds()%60):02d}"

print(f"\n{'='*50}")
print(f"FINAL SESSION REPORT")
print(f"{'='*50}")
print(f"Session Duration:  {duration}")
print(f"Total Inspected:   {total_inspected}")
print(f"Total Passed:      {total_passed}")
print(f"Total Failed:      {total_failed}")
print(f"Final Pass Rate:   {pass_rate:.1f}%")
print(f"Line Quality:      {'GOOD' if pass_rate >= 80 else 'NEEDS REVIEW'}")
print(f"Logs saved to:     {SAVE_DIR}")
print(f"{'='*50}")

cap.release()
cv2.destroyAllWindows()