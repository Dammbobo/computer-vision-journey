from ultralytics import YOLO
import cv2
import urllib.request
import os

# Load model
model = YOLO("yolov8n.pt")

# Download a sample street image
url = "https://ultralytics.com/images/bus.jpg"
img_path = "test_image.jpg"

if not os.path.exists(img_path):
    urllib.request.urlretrieve(url, img_path)
    print("Image downloaded!")

# Load and run detection
img = cv2.imread(img_path)
results = model(img, verbose=False)

# Draw results
annotated = results[0].plot()

# Show result
cv2.imshow("YOLO on Real Image", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()