from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# Define a zone (rectangle coordinates)
ZONE_X1, ZONE_Y1 = 150, 100
ZONE_X2, ZONE_Y2 = 500, 400

# Track total unique people seen
total_count = 0
tracked_ids = set()

print("People Counter running! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the zone on frame
    cv2.rectangle(frame, (ZONE_X1, ZONE_Y1),
                 (ZONE_X2, ZONE_Y2), (255, 255, 0), 2)
    cv2.putText(frame, "MONITORED ZONE", (ZONE_X1, ZONE_Y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Run YOLO tracking
    results = model.track(frame, persist=True, verbose=False)

    # People in zone counter for this frame
    people_in_zone = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().tolist()
        track_ids = results[0].boxes.id.int().tolist()
        labels = results[0].boxes.cls.int().tolist()
        confidences = results[0].boxes.conf.tolist()

        for box, track_id, label, conf in zip(boxes, track_ids, labels, confidences):
            name = model.names[label]

            # Only process people
            if name != "person" or conf < 0.5:
                continue

            x1, y1, x2, y2 = box

            # Get center of person
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if person is inside the zone
            in_zone = (ZONE_X1 < cx < ZONE_X2 and
                      ZONE_Y1 < cy < ZONE_Y2)

            # Count unique people total
            if track_id not in tracked_ids:
                tracked_ids.add(track_id)
                total_count += 1

            # Set color based on zone
            color = (0, 0, 255) if in_zone else (0, 255, 0)

            if in_zone:
                people_in_zone += 1

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"Person #{track_id}",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display stats
    cv2.putText(frame, f"In Zone: {people_in_zone}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Seen: {total_count}",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2)

    # Alert if someone is in zone
    if people_in_zone > 0:
        cv2.putText(frame, "ZONE ALERT!",
                   (ZONE_X1, ZONE_Y2 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 255), 3)

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()