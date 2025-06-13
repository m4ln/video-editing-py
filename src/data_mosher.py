#!/usr/bin/env python3

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

        def in_any_range(idx):
            return any(start <= idx < end for start, end in ranges)

        def write_frame(frame):
            self.out_file.write(frame_start + frame)

        def mosh_delta_repeat(frames, n_repeat, ranges):
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

        def mosh_iframe_removal(frames, ranges):
            for idx, frame in enumerate(frames):
                if in_any_range(idx) and frame[5:8] == iframe:
                    continue  # Remove I-frames in the specified ranges
                write_frame(frame)

        if self.delta:
            mosh_delta_repeat(frames, self.delta, ranges)
        else:
            mosh_iframe_removal(frames, ranges)

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

        n_video_frames = len(
            [frame for frame in frames if frame[5:8] == iframe or frame[5:8] == pframe])
        end_frame = self.end_frame if self.end_frame >= 0 else n_video_frames

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

        if self.delta:
            mosh_delta_repeat(self.delta)
        else:
            mosh_iframe_removal()

        self.export_video()
        self.cleanup()

    def export_video(self):
        subprocess.call(
            f'ffmpeg -loglevel error -y -i {self.output_avi} -crf 18 -pix_fmt yuv420p -vcodec libx264 -acodec aac -b 10000k -r {self.fps} {self.output_video}',
            shell=True
        )

    def process_and_concat(self):
        temp_videos = []
        for idx, (start, end) in enumerate(zip(self.start_frames, self.end_frames)):
            temp_output = os.path.join(
                self.results_dir, f"temp_mosh_{start}_{end}.mp4")
            temp_videos.append(temp_output)
            self.process_segment(start, end, temp_output)

        # Create a file list for ffmpeg concat
        concat_list_path = os.path.join(self.results_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for video in temp_videos:
                f.write(f"file '{video}'\n")

        final_output = os.path.join(
            self.results_dir, f"{self.save_path}_moshed.mp4")
        subprocess.call(
            f"ffmpeg -loglevel error -y -f concat -safe 0 -i {concat_list_path} -c copy {final_output}",
            shell=True
        )

        # Cleanup temp files
        for video in temp_videos:
            if os.path.exists(video):
                os.remove(video)
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)

        print(f"Final video saved to {final_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default='mov03.mov', help='File to be moshed')
    parser.add_argument('--start_frames', nargs='+', type=int,
                        required=True, help='List of start frames')
    parser.add_argument('--end_frames', nargs='+', type=int,
                        required=True, help='List of end frames')
    parser.add_argument('--fps', '-f', default=30, type=int,
                        help='fps to convert initial video to')
    parser.add_argument('--save_path', type=str, default='out',
                        help="Base path to save processed video.")
    parser.add_argument('--delta', '-d', default=0, type=int,
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
    if len(args.start_frames) == 1 and len(args.end_frames) == 1:
        mosher.process_segment(args.start_frames[0], args.end_frames[0], os.path.join(
            mosher.results_dir, f"{args.save_path}_moshed.mp4"))
    else:
        mosher.process_all_ranges()


if __name__ == "__main__":
    main()
