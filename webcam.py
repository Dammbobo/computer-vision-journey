import cv2
import numpy as np

cap = cv2.VideoCapture(0)

print("Webcam started! Hold something RED in front of camera. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Highlight detected red areas on original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Live Camera", frame)
    cv2.imshow("Red Detection Live", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()