import cv2
import numpy as np

# Create a sample image with shapes
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
cv2.circle(img, (300, 300), 80, (255, 255, 255), -1)
cv2.triangle = cv2.polylines

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges
edges = cv2.Canny(gray, 100, 200)

# Show both images side by side
cv2.imshow("Original", img)
cv2.imshow("Edges Detected", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()