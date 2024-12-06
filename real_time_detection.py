# real_time_detection.py

import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("runs/train/biodegradable_detection/weights/best.pt")  # Update path if different

# Initialize webcam (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform inference
    results = model(frame)

    # Parse results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            # Confidence
            conf = box.conf[0]
            # Class ID
            cls = box.cls[0].int()
            # Class name
            class_name = model.names[cls]

            # Define color based on class
            if class_name == 'biodegradable':
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Put label
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
