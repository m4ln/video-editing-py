#!/usr/bin/env python3

import cv2
import os
import argparse
import subprocess


class DataMosher:
    def __init__(self, video, start_frames, end_frames, fps, save_path, delta):
        self.video = video
        self.start_frames = start_frames
        self.end_frames = end_frames
        self.fps = fps
        self.save_path = save_path
        self.delta = delta

        self.video_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', self.video)
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Video file {self.video_path} does not exist in the data directory.")
        self.results_dir = os.path.join(
            os.path.dirname(__file__), '..', 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.input_avi = 'datamoshing_input.avi'
        self.output_avi = 'datamoshing_output.avi'
        self.in_file = None
        self.out_file = None

    def convert_to_avi(self):
        subprocess.call(
            f'ffmpeg -loglevel error -y -i {self.video_path} -crf 0 -pix_fmt yuv420p -bf 0 -b 10000k -r {self.fps} {self.input_avi}',
            shell=True
        )

    def get_fps(self):
        cmd = [
            "ffprobe", "-v", "0", "-of", "csv=p=0",
            "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", self.video_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        num, denom = map(int, output.split('/'))
        return num / denom if denom != 0 else 0

    def open_files(self):
        self.in_file = open(self.input_avi, 'rb')
        self.out_file = open(self.output_avi, 'wb')

    def cleanup(self):
        if self.in_file:
            self.in_file.close()
        if self.out_file:
            self.out_file.close()
        if os.path.exists(self.input_avi):
            os.remove(self.input_avi)
        if os.path.exists(self.output_avi):
            os.remove(self.output_avi)

    def process_all_ranges(self):
        def in_any_range(idx):
            return any(start <= idx < end for start, end in ranges)

        def write_frame(frame):
            self.out_file.write(frame_start + frame)

        def mosh_delta_repeat(frames, n_repeat):
            repeat_frames = []
            repeat_index = 0
            for idx, frame in enumerate(frames):
                if not in_any_range(idx):
                    write_frame(frame)
                    continue
                if (frame[5:8] != iframe and frame[5:8] != pframe):
                    write_frame(frame)
                    continue
                if len(repeat_frames) < n_repeat and frame[5:8] != iframe:
                    repeat_frames.append(frame)
                    write_frame(frame)
                elif len(repeat_frames) == n_repeat:
                    write_frame(repeat_frames[repeat_index])
                    repeat_index = (repeat_index + 1) % n_repeat
                else:
                    write_frame(frame)

        def mosh_iframe_removal(frames):
            for idx, frame in enumerate(frames):
                if in_any_range(idx) and frame[5:8] == iframe:
                    continue  # Remove I-frames in the specified ranges
                write_frame(frame)

        self.convert_to_avi()
        self.open_files()
        in_file_bytes = self.in_file.read()
        frame_start = bytes.fromhex('30306463')
        frames = in_file_bytes.split(frame_start)
        self.out_file = open(os.path.join(
            self.results_dir, f"{self.save_path}_moshed.avi"), 'wb')
        self.out_file.write(frames[0])
        frames = frames[1:]

        iframe = bytes.fromhex('0001B0')
        pframe = bytes.fromhex('0001B6')

        # Prepare a list of (start, end) tuples
        ranges = list(zip(self.start_frames, self.end_frames))

        if self.delta:
            mosh_delta_repeat(frames, self.delta)
        else:
            mosh_iframe_removal(frames)

        self.out_file.close()
        # Export the final video
        output_avi = os.path.join(
            self.results_dir, f"{self.save_path}_moshed.avi")
        final_output = os.path.join(
            self.results_dir, f"{self.save_path}_moshed.mp4")
        subprocess.call(
            f'ffmpeg -loglevel error -y -i {output_avi} -crf 18 -pix_fmt yuv420p -vcodec libx264 -acodec aac -b 10000k -r {self.fps} {final_output}',
            shell=True
        )
        os.remove(output_avi)
        self.cleanup()
        print(f"Final video saved to {final_output}")

    def process_segment(self, start_frame, end_frame, output_video):
        def write_frame(frame):
            self.out_file.write(frame_start + frame)

        def mosh_iframe_removal():
            for index, frame in enumerate(frames):
                if index < self.start_frame or end_frame < index or frame[5:8] != iframe:
                    write_frame(frame)

        def mosh_delta_repeat(n_repeat):
            if n_repeat > end_frame - self.start_frame:
                print('not enough frames to repeat')
                self.cleanup()
                exit(0)
            repeat_frames = []
            repeat_index = 0
            for index, frame in enumerate(frames):
                if (frame[5:8] != iframe and frame[5:8] != pframe) or not self.start_frame <= index < end_frame:
                    write_frame(frame)
                    continue
                if len(repeat_frames) < n_repeat and frame[5:8] != iframe:
                    repeat_frames.append(frame)
                    write_frame(frame)
                elif len(repeat_frames) == n_repeat:
                    write_frame(repeat_frames[repeat_index])
                    repeat_index = (repeat_index + 1) % n_repeat
                else:
                    write_frame(frame)

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output_video = output_video
        self.convert_to_avi()
        self.open_files()
        in_file_bytes = self.in_file.read()
        frame_start = bytes.fromhex('30306463')
        frames = in_file_bytes.split(frame_start)
        self.out_file.write(frames[0])
        frames = frames[1:]

        iframe = bytes.fromhex('0001B0')
        pframe = bytes.fromhex('0001B6')

        self.n_video_frames = len(
            [frame for frame in frames if frame[5:8] == iframe or frame[5:8] == pframe])
        end_frame = self.end_frame if self.end_frame >= 0 else self.n_video_frames

        if self.delta:
            mosh_delta_repeat(self.delta)
        else:
            mosh_iframe_removal()

        self.export_video()
        self.cleanup()

    def process_video(self):
        if len(self.start_frames) == 1 and len(self.end_frames) == 1:
            self.process_segment(self.start_frames[0], self.end_frames[0], os.path.join(
                self.results_dir, f"{self.save_path}_moshed.mp4"))
        else:
            self.process_all_ranges()

    def export_video(self):
        subprocess.call(
            f'ffmpeg -loglevel error -y -i {self.output_avi} -crf 18 -pix_fmt yuv420p -vcodec libx264 -acodec aac -b 10000k -r {self.fps} {self.output_video}',
            shell=True
        )


# helper functions


def display_video_with_frame_counts(video_path):
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}, FPS: {fps}")

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


def main():
    def_video = 'dan_0614.mov'
    def_start_frames = [2, 100]
    def_end_frames = [50, 200]
    def_fps = 30
    def_out = def_video.split('.')[0]
    def_delta = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default=def_video, help='File to be moshed')
    parser.add_argument('--start_frames', nargs='+', type=int,
                        required=False, default=def_start_frames, help='List of start frames (default: [2])')
    parser.add_argument('--end_frames', nargs='+', type=int,
                        required=False, default=def_end_frames, help='List of end frames (default: [-1])')
    parser.add_argument('--fps', '-f', default=def_fps, type=int,
                        help='fps to convert initial video to')
    parser.add_argument('--save_path', type=str, default=def_out,
                        help="Base path to save processed video.")
    parser.add_argument('--delta', '-d', default=def_delta, type=int,
                        help='number of delta frames to repeat')
    args = parser.parse_args()

    if len(args.start_frames) != len(args.end_frames):
        raise ValueError(
            "start_frames and end_frames must have the same length.")

    mosher = DataMosher(
        video=args.video,
        start_frames=args.start_frames,
        end_frames=args.end_frames,
        fps=args.fps,
        save_path=args.save_path,
        delta=args.delta
    )

    # display video with frame counts
    # display_video_with_frame_counts(mosher.video_path)

    # fps check
    print(f"Video FPS: {mosher.get_fps()}")

    # data moshing
    mosher.process_video()

    # print number of total frames
    n_frames = mosher.n_video_frames
    print(f"Total frames count: {n_frames}")


if __name__ == "__main__":
    main()
