

import cv2
import os
import subprocess


def display_video_with_frame_counts(video_path, fps=30):
    """
    Displays a video with frame counts overlayed on each frame.
    Press 'q' to quit, 'p' to pause, 'a' to go back one frame, and 'd' to go forward one frame.

    Args:
        video_path (str): Path to the video file.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Total frames: {frame_count}, FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(frame, f'Frame: {current_frame}/{frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('p'):  # Press 'p' to pause
            cv2.waitKey(-1)  # Wait indefinitely until any key is pressed
        elif key == ord('a'):  # Press 'a' to go back one frame
            pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        elif key == ord('d'):  # Press 'd' to go forward one frame
            pos = min(frame_count - 1, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    cap.release()
    cv2.destroyAllWindows()


def crop_video(video_path, start_frame, end_frame):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    if start_frame < 0 or end_frame < 0 or start_frame >= end_frame:
        raise ValueError("Invalid start or end frame values.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}, FPS: {fps}")

    # Set the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Get the codec information
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base, ext = os.path.splitext(video_path)
    out_filename = f"{base}_cropped_{start_frame}_{end_frame}{ext}"
    out = cv2.VideoWriter(out_filename, fourcc, cap.get(
        cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
    while True:
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video cropped and saved as: {out_filename}")


def export_video_with_frame_counts(video_path):
    base, ext = os.path.splitext(video_path)
    out_filename = f"{base}_with_counts{ext}"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    subprocess.call(
        f'ffmpeg -i {video_path} -vf "drawtext=fontfile=Arial.ttf: text=%{{n}}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099: fontsize=72" {out_filename}', shell=True)


if __name__ == "__main__":
    video = 'dan_0614_cropped_0_1000.mov'  # Example video file name
    video_path = os.path.join(os.path.dirname(
        __file__), '..', 'data', 'dance', video)

    # display video with frame counts
    # display_video_with_frame_counts(video_path, fps=30)

    # crop video
    # crop_video(video_path, 1000, 2000)

    # export video with frame counts
    export_video_with_frame_counts(video_path)
