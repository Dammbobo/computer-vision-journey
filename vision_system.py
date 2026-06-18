import cv2
import numpy as np

cap = cv2.VideoCapture(0)

print("Vision System Running! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert to grayscale for shape detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Define red color range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours of red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter small noise — only detect objects bigger than 500 pixels
        if area > 500:
            # Get shape
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            if len(approx) == 3:
                shape = "Triangle"
                color = (0, 255, 255)
            elif len(approx) == 4:
                shape = "Rectangle"
                color = (255, 0, 0)
            else:
                shape = "Circle"
                color = (0, 255, 0)

            # Draw outline
            cv2.drawContours(frame, [cnt], -1, color, 2)

            # Get center position
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Label shape and area
                cv2.putText(frame, f"{shape}", (cx - 40, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Area: {int(area)}px", (cx - 40, cy + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display object count
    cv2.putText(frame, f"Objects detected: {len(contours)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Vision System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()