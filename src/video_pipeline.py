import cv2
import os

from src.yolo_detector import YOLOVideoDetector

class ViedeoPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detector = None
        self.datamosher = None
        self.save_path = None
        self.frame_skip = 2  # Default frame skip value

    def add_yolo_detector(self, model_type="yolov3", confidence_threshold=0.5, nms_threshold=0.4):
        self.detector = YOLOVideoDetector(model_type, confidence_threshold, nms_threshold)
    
    def add_data_mosher():
        pass

    def set_save_path(self, save_path):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def process_video(self, frame_skip=2):
        self.frame_skip = frame_skip
    
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:  # Skip frames based on frame_skip
                continue
            
            # Process the frame
            if self.detector is not None:
                # Use the detector to process the frame
                frame = self.detector.process_frame(frame)
            if self.datamosher is not None:
                # Use the datamosher to process the frame   
                frame = self.datamosher.process_frame(frame)
            
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()