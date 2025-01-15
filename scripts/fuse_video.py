import subprocess
from pathlib import Path
import argparse


def fuse_videos(directory):
    directory = Path(directory)
    for subdir in directory.iterdir():
        if subdir.is_dir():
            flow_videos = []
            rgb_videos = []

            for file in subdir.iterdir():
                if file.suffix == ".mp4":
                    if "flow" in file.name:
                        flow_videos.append(file)
                    elif "rgb" in file.name:
                        rgb_videos.append(file)

            parent_name = subdir.name

            if flow_videos:
                flow_videos.sort()
                flow_output = subdir / f"0flow_{parent_name}.mp4"
                fuse_video_files(flow_videos, flow_output)

            if rgb_videos:
                rgb_videos.sort()
                rgb_output = subdir / f"0rgb_{parent_name}.mp4"
                fuse_video_files(rgb_videos, rgb_output)


def fuse_video_files(video_files, output_file):
    list_file = Path("input_videos.txt")

    with list_file.open("w") as f:
        for video in video_files:
            f.write(f"file '{video}'\n")

    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_file)
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Fused video saved as: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during fusion: {e}")
    finally:
        list_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse videos by type in each subdirectory.")
    parser.add_argument("directory", type=str,
                        help="Base directory containing subdirectories with videos.")
    args = parser.parse_args()

    base_directory = args.directory
    fuse_videos(base_directory)
