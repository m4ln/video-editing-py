#!/usr/bin/env python3

import os
import argparse
import subprocess

class DataMosher:
    def __init__(self, video, start_frame, end_frame, fps, save_path, delta):
        self.video = video
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps
        self.save_path = save_path
        self.delta = delta

        self.video_path = os.path.join(os.path.dirname(__file__), '..', 'data', self.video)
        self.output_video = os.path.join(os.path.dirname(__file__), '..', 'results', self.save_path)
        if not self.output_video.endswith('.mp4'):
            self.output_video += '.mp4'
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file {self.video_path} does not exist in the data directory.")
        if not os.path.exists(os.path.dirname(self.output_video)):
            os.makedirs(os.path.dirname(self.output_video))

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

    def process(self):
        self.convert_to_avi()
        self.open_files()
        in_file_bytes = self.in_file.read()
        frame_start = bytes.fromhex('30306463')
        frames = in_file_bytes.split(frame_start)
        self.out_file.write(frames[0])
        frames = frames[1:]

        iframe = bytes.fromhex('0001B0')
        pframe = bytes.fromhex('0001B6')

        n_video_frames = len([frame for frame in frames if frame[5:8] == iframe or frame[5:8] == pframe])
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='hand.mov', help='File to be moshed')
    parser.add_argument('--start_frame', '-s', default=10, type=int, help='start frame of the mosh')
    parser.add_argument('--end_frame', '-e', default=100, type=int, help='end frame of the mosh')
    parser.add_argument('--fps', '-f', default=30, type=int, help='fps to convert initial video to')
    parser.add_argument('--save_path', type=str, default='out', help="Path to save the processed video.")
    parser.add_argument('--delta', '-d', default=0, type=int, help='number of delta frames to repeat')
    args = parser.parse_args()

    mosher = DataMosher(
        video=args.video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        save_path=args.save_path,
        delta=args.delta
    )
    mosher.process()

if __name__ == "__main__":
    main()