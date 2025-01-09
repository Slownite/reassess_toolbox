import argparse
from pathlib import Path
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Resize videos to the specified resolution using ffmpeg.")
    parser.add_argument("root_dir", type=Path,
                        help="Root directory to search for videos.")
    parser.add_argument("--width", type=int, default=224,
                        help="Width of the resized video.")
    parser.add_argument("--height", type=int, default=224,
                        help="Height of the resized video.")
    return parser.parse_args()


def resize_video(video_path, width, height):
    temp_file = video_path.with_suffix(".temp.mp4")
    cmd = [
        "ffmpeg", "-i", str(video_path), "-vf",
        f"scale={width}:{height}", "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        str(temp_file)
    ]
    subprocess.run(cmd, check=True)
    temp_file.replace(video_path)


def process_videos(root_dir, width, height):
    for video_path in root_dir.rglob("*.mp4"):
        print(f"Resizing {video_path} to {width}x{height}...")
        resize_video(video_path, width, height)


if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.root_dir, args.width, args.height)
