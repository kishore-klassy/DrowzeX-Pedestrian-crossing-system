import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define lane region (adjust based on camera angle)
lane_x_start, lane_x_end = 200, 450
lane_y_start, lane_y_end = 200, 400

# Function to check if an object is within the lane
def is_in_lane(x, y, w, h):
    center_x, center_y = x + w // 2, y + h // 2
    return lane_x_start < center_x < lane_x_end and lane_y_start < center_y < lane_y_end

# Class name mapping for YOLOv8 (COCO)
class_names = {
    0: "Person",
    16: "Dog",
    17: "Cat"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    pedestrian_count = 0  # Counter for pedestrians/animals in lane

    for result in results:
        for box in result.boxes.data:
            x, y, w, h = map(int, box[:4])  # Bounding box
            conf = float(box[4])            # Confidence score
            cls = int(box[5])               # Class ID
            
            if cls in [0, 16, 17]:  # 0: Person, 16: Dog, 17: Cat
                label = class_names.get(cls, "Unknown")
                if is_in_lane(x, y, w, h):
                    pedestrian_count += 1
                    color = (0, 0, 255)  # Red for danger
                else:
                    color = (0, 255, 0)  # Green for safe
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Glow red light if pedestrian/animal is detected
    if pedestrian_count > 0:
        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)  # Red light
    else:
        cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)  # Green light

    # Display pedestrian count
    cv2.putText(frame, f"Pedestrian Count: {pedestrian_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Pedestrian Crossing Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
