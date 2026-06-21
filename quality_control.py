import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Quality standards
GOOD_COLOR_LOWER = np.array([35, 50, 50])
GOOD_COLOR_UPPER = np.array([85, 255, 255])
MIN_AREA = 1000
MAX_AREA = 50000
EXPECTED_CORNERS = 4

# Production counters
total_inspected = 0
total_passed = 0
total_failed = 0
last_status = None
status_timer = 0

print("Quality Control System Running! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, GOOD_COLOR_LOWER, GOOD_COLOR_UPPER)

    # Find contours
    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    current_status = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        approx = cv2.approxPolyDP(
            cnt,
            0.04 * cv2.arcLength(cnt, True),
            True
        )
        corners = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        aspect_ratio = w / h

        defects = []

        if corners != EXPECTED_CORNERS:
            defects.append(f"Shape defect: {corners} corners")
        if area < MIN_AREA * 1.5:
            defects.append("Size defect: too small")
        elif area > MAX_AREA * 0.8:
            defects.append("Size defect: too large")
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            defects.append("Shape defect: deformed")

        if len(defects) == 0:
            current_status = "PASS"
            color = (0, 255, 0)
        else:
            current_status = "FAIL"
            color = (0, 0, 255)

        # Draw detection
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        cv2.circle(display, (cx, cy), 5, color, -1)
        cv2.putText(display, current_status,
                   (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for i, defect in enumerate(defects):
            cv2.putText(display, defect,
                       (x, y + h + 20 + (i * 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 255), 1)

    # Update counters when status changes
    if current_status is not None and current_status != last_status:
        total_inspected += 1
        if current_status == "PASS":
            total_passed += 1
        else:
            total_failed += 1
        last_status = current_status
        status_timer = 30

    # Calculate pass rate
    pass_rate = (total_passed / total_inspected * 100) if total_inspected > 0 else 0

    # Draw dashboard panel
    cv2.rectangle(display, (0, 0), (280, 160), (0, 0, 0), -1)
    cv2.rectangle(display, (0, 0), (280, 160), (255, 255, 255), 1)

    # Display stats
    cv2.putText(display, "QUALITY CONTROL", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Inspected: {total_inspected}",
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, f"Passed: {total_passed}",
               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(display, f"Failed: {total_failed}",
               (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(display, f"Pass Rate: {pass_rate:.1f}%",
               (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
               (0, 255, 0) if pass_rate >= 80 else (0, 0, 255), 1)

    cv2.imshow("Quality Control System", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()