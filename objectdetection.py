import cv2
import numpy as np
from sort import Sort  # Sort is a simple object tracking algorithm
from deep_sort import DeepSort  # DeepSort is a deep learning-based object tracking algorithm

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize Deep SORT object tracker
deepsort = DeepSort("deep_sort_model.pth", max_dist=0.3, min_confidence=0.3, nms_max_overlap=0.5)

# Function to detect and track objects in a video stream
def detect_and_track_objects(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Detect objects in the frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process detection results
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform non-maximum suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Track detected objects using Deep SORT
        if len(boxes) > 0:
            detections = np.array([[x, y, x + w, y + h, conf] for (x, y, w, h), conf in zip(boxes, confidences)])
            tracked_objects = deepsort.update(detections)
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = obj.astype(int)
                label = f"Object {obj_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run object detection and tracking on a video file
if _name_ == "_main_":
    detect_and_track_objects("test_video.mp4")