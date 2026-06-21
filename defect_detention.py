import cv2
import numpy as np

cap = cv2.VideoCapture(0)

print("Defect Detection System Running! Press 'q' to quit")

# Define acceptable color range (we'll use green as our "good" product color)
GOOD_COLOR_LOWER = np.array([35, 50, 50])
GOOD_COLOR_UPPER = np.array([85, 255, 255])

# Define acceptable size range in pixels
MIN_AREA = 1000
MAX_AREA = 50000

# Define acceptable shape (number of corners)
EXPECTED_CORNERS = 4  # We expect rectangular products

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy for display
    display = frame.copy()

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for good color
    color_mask = cv2.inRange(hsv, GOOD_COLOR_LOWER, GOOD_COLOR_UPPER)

    # Find contours of detected objects
    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter by size
        if area < MIN_AREA or area > MAX_AREA:
            continue

        # Get shape
        approx = cv2.approxPolyDP(
            cnt,
            0.04 * cv2.arcLength(cnt, True),
            True
        )
        corners = len(approx)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Get center
        cx = x + w // 2
        cy = y + h // 2

        # Check for defects
        defects = []

        # Shape defect check
        if corners != EXPECTED_CORNERS:
            defects.append(f"Shape defect: {corners} corners")

        # Size defect check
        if area < MIN_AREA * 1.5:
            defects.append("Size defect: too small")
        elif area > MAX_AREA * 0.8:
            defects.append("Size defect: too large")

        # Aspect ratio check (width vs height)
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            defects.append(f"Shape defect: deformed")

        # Determine pass or fail
        if len(defects) == 0:
            status = "PASS"
            color = (0, 255, 0)  # Green
        else:
            status = "FAIL"
            color = (0, 0, 255)  # Red

        # Draw bounding box
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

        # Draw center point
        cv2.circle(display, (cx, cy), 5, color, -1)

        # Display status
        cv2.putText(display, status, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display defects if any
        for i, defect in enumerate(defects):
            cv2.putText(display, defect, (x, y + h + 20 + (i * 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display area and corners
        cv2.putText(display, f"Area: {int(area)}px Corners: {corners}",
                   (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display instructions
    cv2.putText(display, "Hold a GREEN object to test",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2)

    cv2.imshow("Defect Detection System", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()