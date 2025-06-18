# Python Video Editing Tools: YOLO Detection & Datamoshing

This project is a collection of Python-based video editing tools, currently featuring:

- **YOLO Object Detection**: Detect and visualize objects in videos using YOLOv3 or YOLOv4-tiny.
- **Glitching / Datamoshing Effects**: Apply datamoshing effects to your videos.

## Features

- Run YOLO object detection on video files.
- Apply various glitch and datamoshing effects, including I-frame removal, P-frame duplication, and vector motion manipulation.
- Modular design: add your own effects or detection modules easily.

## Installation

1. **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Install FFmpeg (required for datamoshing):**
    - On macOS:
        ```sh
        brew update
        brew upgrade
        brew install ffmpeg
        ffmpeg -version
        ```
    - Or download from [ffmpeg.org](https://ffmpeg.org/).

3. **Install ffedit and ffgac for vector motion and style transfer:**
    - Download from [ffglitch.org](https://ffglitch.org/).
    - Place the binaries in the `bin/` directory.

4. **Download YOLO model files:**
    - Place `yolov3.cfg`, `yolov3.weights`, `yolov4-tiny.cfg`, `yolov4-tiny.weights`, and `coco.names` in the `models/` directory.

## Directory Structure

```
.
├── data/           # Input video files
├── models/         # YOLO model files
├── results/        # Output videos
├── src/            # Source code (pipeline, detectors, datamosher, etc.)
├── requirements.txt
└── README.md
```

## Running Examples

All scripts are in the `src/` directory. Below are example usages:

### 1. YOLO Object Detection

Run YOLO detection on a video:

```sh
python src/video_pipeline.py --video <your_video.mov> --yolo --model yolov3 --frame_skip 2
```

**Options:**
- `--video`: Name of the video file in the `data/` directory (default: `mov04.mov`)
- `--frame_skip`: Number of frames to skip during processing (default: 2)
- `--save_path`: Path to save the processed video (optional)
- `--yolo`: Enable YOLO object detection
- `--model`: YOLO model type (`yolov3` or `yolov4-tiny`)
- `--confidence`: Confidence threshold for detections (default: 0.1)
- `--nms`: Non-Maximum Suppression threshold (default: 0.4)

### 2. Datamoshing / Glitch Effects

#### I-frame Removal or P-frame Duplication

```sh
python src/data_mosher.py --video <your_video.mov> --start_frames 2 --end_frames 100 --delta 30
```
- `--delta`: Number of delta frames to repeat (for P-frame duplication). Omit for I-frame removal.

#### Vector Motion Glitching

```sh
python src/vector_motion.py <input_video> -s src/styles/horizontal_motion_example.py -o <output_video.mp4>
```
- Use different scripts in `src/styles/` for different effects.

#### Style Transfer (Motion Vectors)

Extract motion vectors and transfer them to another video:

```sh
python src/style_transfer.py -e <source_video.mp4> -t <target_video.mp4> <output_video.mp4>
```

## Notes

- Place your input videos in the `data/` directory.
- Output videos will be saved to the `results/` directory.
- You can add new effects by creating scripts in `src/styles/` in either JavaScript or Python.

## License

GNU General Public License v3.0

---

## References

This project builds on and adapts code and concepts from the following repositories:

- [datamoshing](https://github.com/tiberiuiancu/datamoshing)

Please see these repositories for more information and original implementations.