from ultralytics import YOLO
import cv2

# Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# Run detection on your webcam
cap = cv2.VideoCapture(0)

print("YOLO running! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()