import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)

# Create a simple black image with a white circle
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.circle(img, (200, 200), 100, (255, 255, 255), -1)

# Show it
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()