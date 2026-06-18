import cv2
import numpy as np

# Create image with different shapes
img = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.circle(img, (100, 100), 80, (255, 255, 255), -1)
cv2.rectangle(img, (250, 50), (450, 200), (255, 255, 255), -1)
cv2.ellipse(img, (250, 400), (100, 60), 0, 0, 360, (255, 255, 255), -1)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find contours
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw and label each shape
for cnt in contours:
    # Get shape name
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        shape = "Rectangle"
    else:
        shape = "Circle"
    
    # Draw contour outline
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    
    # Get center position for label
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(img, shape, (cx-30, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Shape Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()