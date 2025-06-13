import argparse
import cv2
import os

from yolo_detector import YOLOVideoDetector


class ViedeoPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.detector = None
        self.datamosher = None
        self.save_path = None
        self.frame_skip = 2  # Default frame skip value

    def add_yolo_detector(self, model_type="yolov3", confidence_threshold=0.5, nms_threshold=0.4):
        self.detector = YOLOVideoDetector(
            model_type, confidence_threshold, nms_threshold)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video using different methods.")
    parser.add_argument("--video", type=str, default="mov04.mov",
                        help="Video file to process.")
    parser.add_argument("--frame_skip", type=int, default=2,
                        help="Number of frames to skip during processing.")
    parser.add_argument("--save_path", type=str, default="",
                        help="Path to save the processed video.")
    parser.add_argument("--yolo", action='store_false',
                        help="Enable YOLO object detection.")
    parser.add_argument("--model", type=str, default="yolov3", choices=["yolov3", "yolov4-tiny"],
                        help="Model type to use for detection.")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections.")
    parser.add_argument("--nms", type=float, default=0,
                        help="Non-Maximum Suppression threshold.")
    parser.add_argument("--datamoshing", action='store_true',
                        help="Enable datamoshing effect.")
    args = parser.parse_args()

    # Ensure the video directory and file exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Video data directory does not exist.")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'models')):
        raise FileNotFoundError("Model directory does not exist.")

    video_path = os.path.join(data_dir, args.video)
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file {args.video} does not exist in the data directory.")
    pipeline = ViedeoPipeline(video_path)
    if args.yolo:
        pipeline.add_yolo_detector(
            model_type=args.model, confidence_threshold=args.confidence, nms_threshold=args.nms)
    if args.save_path:
        pipeline.set_save_path(args.save_path)
    pipeline.process_video(frame_skip=args.frame_skip)
    if args.save_path:
        print(f"Processed video saved to {args.save_path}")
