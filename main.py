import cv2
import numpy as np
import os


class YOLOVideoDetector:
    def __init__(self, model_type="yolov3", confidence_threshold=0.5, nms_threshold=0.4):
        self.model_path = os.path.join(os.path.dirname(__file__), 'models')
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Load YOLO model
        self.net = self.load_model(model_type)

        # Get output layer names
        layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        self.output_layers = [layer_names[i - 1] for i in unconnected_layers]

        # Load class names
        coco_names_path = os.path.join(self.model_path, "coco.names")
        with open(coco_names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def load_model(self, model_type):
        if model_type == "yolov3":
            weights = os.path.join(self.model_path, "yolov3.weights")
            config = os.path.join(self.model_path, "yolov3.cfg")
        elif model_type == "yolov4-tiny":
            weights = os.path.join(self.model_path, "yolov4-tiny.weights")
            config = os.path.join(self.model_path, "yolov4-tiny.cfg")
        else:
            raise ValueError("Unsupported model type. Choose 'yolov3' or 'yolov4-tiny'.")
        return cv2.dnn.readNet(weights, config)

    def process_video(self, filename):
        video_path = os.path.join(self.data_dir, filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video file {filename}")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:  # Skip every other frame
                continue

            processed_frame = self.process_frame(frame)
            cv2.imshow("Video", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        frame_width = frame.shape[1] // 3
        frame_height = frame.shape[0] // 3
        frame = cv2.resize(frame, (frame_width, frame_height))
        height, width, _ = frame.shape

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Process detections
        class_ids, confidences, boxes = self.get_detections(outs, width, height)

        # Non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        # Draw bounding boxes
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def get_detections(self, outs, width, height):
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return class_ids, confidences, boxes


if __name__ == "__main__":
    detector = YOLOVideoDetector(model_type="yolov3")
    detector.process_video("mov00.mov")