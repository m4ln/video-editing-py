import argparse
import os

from src.video_pipeline import ViedeoPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video using different methods.")
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
    parser.add_argument("--nms", type=float, default=0.4,
                        help="Non-Maximum Suppression threshold.")
    parser.add_argument("--datamoshing", action='store_true',
                        help="Enable datamoshing effect.")
    args = parser.parse_args()
    # Ensure the video directory and file exist
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data')):
        raise FileNotFoundError("Video data directory does not exist.")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        raise FileNotFoundError("Model directory does not exist.")

    video_path = os.path.join(os.path.dirname(__file__), 'data', args.video)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {args.video} does not exist in the data directory.")
    pipeline = ViedeoPipeline(video_path)
    if args.yolo:
        pipeline.add_yolo_detector(model_type=args.model, confidence_threshold=args.confidence, nms_threshold=args.nms)
    if args.save_path:
        pipeline.set_save_path(args.save_path)
    pipeline.process_video(frame_skip=args.frame_skip)
    if args.save_path:
        print(f"Processed video saved to {args.save_path}")