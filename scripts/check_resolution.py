import argparse
from pathlib import Path
import subprocess
import csv


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Check video resolutions and output to CSV.")
    parser.add_argument("root_dir", type=Path,
                        help="Root directory to search for videos.")
    parser.add_argument("output_csv", type=Path,
                        help="Path to the output CSV file.")
    return parser.parse_args()


def get_video_resolution(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height", "-of", "csv=p=0", str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=True)
    width, height = map(int, result.stdout.strip().split(','))
    return width, height


def process_videos(root_dir, output_csv):
    videos_data = []

    for video_path in root_dir.rglob("*.mp4"):
        try:
            width, height = get_video_resolution(video_path)
            videos_data.append((video_path.name, width, height))
            print(f"Processed {video_path}: {width}x{height}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    # Write to CSV
    with output_csv.open(mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Video Name", "Width", "Height"])
        writer.writerows(videos_data)


if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.root_dir, args.output_csv)
