# YOLO Video Detection & Datamoshing Pipeline

This project provides a flexible pipeline for video processing using YOLO object detection and datamoshing effects. You can combine, extend, and customize video processing steps for creative or analytical workflows.

## Features

- **YOLO Object Detection**: Detect objects in videos using YOLOv3 or YOLOv4-tiny.
- **Datamoshing**: Apply glitch-art datamoshing effects to your videos.
- **Pipeline Architecture**: Easily combine multiple video processing methods.

## Installation

1. **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Install FFmpeg (required for video processing):**

    On macOS, using Homebrew:
    ```sh
    brew update
    brew upgrade
    brew install ffmpeg
    ```

    Check your FFmpeg installation:
    ```sh
    ffmpeg -version
    ```

## Usage

### Run the Main Pipeline

```sh
python main.py --video <your_video.mov> [--yolo] [--datamoshing] [other options]
```

**Options:**
- `--video`: Name of the video file in the `data/` directory (default: `mov04.mov`)
- `--frame_skip`: Number of frames to skip during processing (default: 2)
- `--save_path`: Path to save the processed video
- `--yolo`: Enable YOLO object detection
- `--model`: YOLO model type (`yolov3` or `yolov4-tiny`)
- `--confidence`: Confidence threshold for detections (default: 0.5)
- `--nms`: Non-Maximum Suppression threshold (default: 0.4)
- `--datamoshing`: Enable datamoshing effect

### Example

```sh
python main.py --video sample.mov --yolo --model yolov3 --datamoshing --save_path results/output.mp4
```

## Directory Structure

```
.
├── data/           # Input video files
├── models/         # YOLO model files
├── results/        # Output videos
├── src/            # Source code (pipeline, detectors, datamosher)
├── main.py         # Main entry point
├── requirements.txt
└── README.md
```

## Notes

- Ensure your input videos are placed in the `data/` directory.
- Output videos will be saved to the `results/` directory.
- You can extend the pipeline by adding new processing classes in `src/`.

## License

GNU License

---