from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Only detect these objects
TARGET_OBJECTS = ["person", "cup", "keyboard", "mouse", "chair"]

cap = cv2.VideoCapture(0)

print("Filtered YOLO running! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    # Get detections
    detections = results[0].boxes

    # Count objects
    object_counts = {}

    for box in detections:
        # Get object name
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])

        # Only process target objects
        if label in TARGET_OBJECTS and confidence > 0.5:
            # Count it
            object_counts[label] = object_counts.get(label, 0) + 1

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label and confidence
            cv2.putText(frame, f"{label} {confidence:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)

    # Display counts on screen
    y_pos = 30
    for obj, count in object_counts.items():
        cv2.putText(frame, f"{obj}: {count}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30

    cv2.imshow("Filtered YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()