from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

print("Object Tracking running! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with tracking enabled
    results = model.track(frame, persist=True, verbose=False)

    # Get detections
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().tolist()
        track_ids = results[0].boxes.id.int().tolist()
        labels = results[0].boxes.cls.int().tolist()
        confidences = results[0].boxes.conf.tolist()

        for box, track_id, label, conf in zip(boxes, track_ids, labels, confidences):
            x1, y1, x2, y2 = box
            name = model.names[label]

            # Give each ID a unique color
            color = (
                (track_id * 50) % 255,
                (track_id * 100) % 255,
                (track_id * 150) % 255
            )

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label with tracking ID
            cv2.putText(frame, f"{name} #{track_id} {conf:.2f}",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()