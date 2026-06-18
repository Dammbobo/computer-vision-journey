import cv2
import numpy as np

# Create an image with colored shapes
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.circle(img, (100, 100), 80, (0, 0, 255), -1)    # Red circle
cv2.rectangle(img, (200, 200), (350, 350), (0, 255, 0), -1)  # Green rectangle
cv2.circle(img, (300, 100), 60, (255, 0, 0), -1)    # Blue circle

# Convert to HSV (better for color detection)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define red color range and create a mask
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Red Detection Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()