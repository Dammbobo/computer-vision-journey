import cv2
import numpy as np
import os
from datetime import datetime

cap = cv2.VideoCapture(0)

# Quality standards
GOOD_COLOR_LOWER = np.array([0, 50, 50])
GOOD_COLOR_UPPER = np.array([180, 255, 255])
MIN_AREA = 1000
MAX_AREA = 50000
EXPECTED_CORNERS = 4

# Production counters
total_inspected = 0
total_passed = 0
total_failed = 0
last_status = None

# Save directory
SAVE_DIR = "quality_control_logs"
DEFECT_DIR = os.path.join(SAVE_DIR, "defects")
PASS_DIR = os.path.join(SAVE_DIR, "passed")
os.makedirs(DEFECT_DIR, exist_ok=True)
os.makedirs(PASS_DIR, exist_ok=True)

# Session start time
session_start = datetime.now()

print("HMI Dashboard Running! Press 'q' to quit")

def draw_dashboard(frame, total_inspected, total_passed, 
                   total_failed, current_status, session_start):
    
    h, w = frame.shape[:2]
    
    # Create dashboard panel on right side
    panel_width = 280
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)  # Dark background
    
    # Header
    cv2.rectangle(panel, (0, 0), (panel_width, 60), (40, 40, 40), -1)
    cv2.putText(panel, "INDUSTRIAL AI", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(panel, "QUALITY CONTROL", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    
    # Divider line
    cv2.line(panel, (0, 62), (panel_width, 62), (60, 60, 60), 1)
    
    # Current status box
    status_color = (0, 255, 0) if current_status == "PASS" else \
                   (0, 0, 255) if current_status == "FAIL" else \
                   (100, 100, 100)
    status_text = current_status if current_status else "SCANNING"
    
    cv2.rectangle(panel, (10, 75), (panel_width-10, 130),
                 status_color, -1)
    cv2.putText(panel, status_text,
               (panel_width//2 - len(status_text)*10, 112),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Stats section
    cv2.putText(panel, "PRODUCTION STATS", (10, 155),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.line(panel, (10, 162), (panel_width-10, 162), (60, 60, 60), 1)
    
    # Inspected
    cv2.putText(panel, "Inspected", (10, 185),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, str(total_inspected), (panel_width-60, 185),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Passed
    cv2.putText(panel, "Passed", (10, 215),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, str(total_passed), (panel_width-60, 215),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Failed
    cv2.putText(panel, "Failed", (10, 245),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, str(total_failed), (panel_width-60, 245),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Pass rate
    pass_rate = (total_passed / total_inspected * 100) \
                if total_inspected > 0 else 0
    rate_color = (0, 255, 0) if pass_rate >= 80 else (0, 0, 255)
    
    cv2.line(panel, (10, 260), (panel_width-10, 260), (60, 60, 60), 1)
    cv2.putText(panel, "Pass Rate", (10, 285),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"{pass_rate:.1f}%", (panel_width-80, 285),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, rate_color, 2)
    
    # Pass rate bar
    bar_width = panel_width - 20
    cv2.rectangle(panel, (10, 295), (panel_width-10, 315),
                 (60, 60, 60), -1)
    fill = int(bar_width * pass_rate / 100)
    cv2.rectangle(panel, (10, 295), (10 + fill, 315),
                 rate_color, -1)
    
    # Session time
    elapsed = datetime.now() - session_start
    minutes = int(elapsed.total_seconds() // 60)
    seconds = int(elapsed.total_seconds() % 60)
    
    cv2.line(panel, (10, 330), (panel_width-10, 330), (60, 60, 60), 1)
    cv2.putText(panel, "Session Time", (10, 355),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, f"{minutes:02d}:{seconds:02d}",
               (panel_width-80, 355),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Current time
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(panel, "Time", (10, 385),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, current_time, (panel_width-100, 385),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Footer
    cv2.rectangle(panel, (0, h-40), (panel_width, h),
                 (40, 40, 40), -1)
    cv2.putText(panel, "AI Vision System v1.0", (10, h-15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Combine camera feed and dashboard panel
    combined = np.hstack([frame, panel])
    return combined


while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, GOOD_COLOR_LOWER, GOOD_COLOR_UPPER)

    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    current_status = None
    current_defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        approx = cv2.approxPolyDP(
            cnt, 0.04 * cv2.arcLength(cnt, True), True)
        corners = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        aspect_ratio = w / h

        defects = []
        if corners != EXPECTED_CORNERS:
            defects.append(f"Shape-{corners}corners")
        if area < MIN_AREA * 1.5:
            defects.append("Size-small")
        elif area > MAX_AREA * 0.8:
            defects.append("Size-large")
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            defects.append("Shape-deformed")

        if len(defects) == 0:
            current_status = "PASS"
            color = (0, 255, 0)
        else:
            current_status = "FAIL"
            color = (0, 0, 255)
            current_defects = defects

        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        cv2.circle(display, (cx, cy), 5, color, -1)
        cv2.putText(display, current_status, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for i, defect in enumerate(defects):
            cv2.putText(display, defect,
                       (x, y + h + 20 + (i * 20)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 255), 1)

    # Save images when status changes
    if current_status is not None and current_status != last_status:
        total_inspected += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if current_status == "PASS":
            total_passed += 1
            filename = f"PASS_{timestamp}.jpg"
            cv2.imwrite(os.path.join(PASS_DIR, filename), display)
        else:
            total_failed += 1
            defect_text = "_".join(current_defects)
            filename = f"FAIL_{defect_text}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(DEFECT_DIR, filename), display)

        last_status = current_status

    # Draw full HMI dashboard
    output = draw_dashboard(
        display, total_inspected, total_passed,
        total_failed, current_status, session_start
    )

    cv2.imshow("Industrial AI Quality Control System", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Session report
pass_rate = (total_passed / total_inspected * 100) \
            if total_inspected > 0 else 0
print(f"\n{'='*40}")
print(f"SESSION REPORT")
print(f"{'='*40}")
print(f"Total Inspected: {total_inspected}")
print(f"Total Passed:    {total_passed}")
print(f"Total Failed:    {total_failed}")
print(f"Pass Rate:       {pass_rate:.1f}%")
print(f"Logs saved to:   {SAVE_DIR}")
print(f"{'='*40}")

cap.release()
cv2.destroyAllWindows()