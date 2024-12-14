from ultralytics import YOLO
import cv2
import math 

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes (YOLO class names)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Index of the "person" class
PERSON_CLASS_ID = classNames.index("person")

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform detection
    results = model(img, stream=True)

    # Process results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Get class ID
            cls = int(box.cls[0])

            # Only process "person" class
            if cls == PERSON_CLASS_ID:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int values

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence score
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Display class name and confidence
                label = f"{classNames[cls]} {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # Show the frame
    cv2.imshow('Webcam', img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()