import cv2
import numpy as np
import os
from threading import Thread

# def process_frame():
#     while True:
#         # Process frames here
#         pass

# Thread(target=process_frame).start()

# Load YOLO
model_path = os.path.join(os.path.dirname(__file__), 'models')
net = cv2.dnn.readNet(
    os.path.join(model_path, "yolov3.weights"),
    os.path.join(model_path, "yolov3.cfg")
)

net = cv2.dnn.readNet(
    os.path.join(model_path, "yolov4-tiny.weights"),
    os.path.join(model_path, "yolov4-tiny.cfg")
)

# get output layer names
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_layers]

# Load class names
coco_names_path = os.path.join(model_path, "coco.names")
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video file
data_dir = os.path.join(os.path.dirname(__file__), 'data')
filename = 'mov00.mov'
cap = cv2.VideoCapture(os.path.join(data_dir, filename))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 1 != 0:  # Skip every other frame
        continue

    # Prepare the frame for YOLO
    frame_width = int(cap.get(3))//2
    frame_height = int(cap.get(4))//2
    frame = cv2.resize(frame, (frame_width, frame_height))
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()