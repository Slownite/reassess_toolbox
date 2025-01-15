from pathlib import Path
import subprocess
import argparse


def process_video_subdirectories(root_dir: str):
    """
    Process subdirectories containing 'rgb_*.mp4' or 'flow_*.mp4' files by fusing them together.

    Parameters:
    - root_dir (str): The root directory to search for subdirectories.
    """
    root_path = Path(root_dir)

    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(
            f"The provided root directory does not exist or is not a directory: {root_dir}")

    for sub_dir in root_path.iterdir():
        if sub_dir.is_dir():
            # Find files matching the patterns
            rgb_files = list(sub_dir.glob("rgb_*.mp4"))
            flow_files = list(sub_dir.glob("flow_*.mp4"))

            if rgb_files:
                output_file = sub_dir / f"rgb_{sub_dir.name}.mp4"
                fuse_videos(output_file, rgb_files)

            if flow_files:
                output_file = sub_dir / f"flow_{sub_dir.name}.mp4"
                fuse_videos(output_file, flow_files)


def fuse_videos(output_file: Path, input_files: list):
    """
    Fuse multiple video files together into a single output file using ffmpeg.

    Parameters:
    - output_file (Path): The output .mp4 file.
    - input_files (list): List of input .mp4 files.
    """
    if len(input_files) == 1:
        # If there's only one file, simply copy it to the output
        input_file = input_files[0]
        print(f"Only one video found. Copying {input_file} to {output_file}.")
        subprocess.run(["cp", str(input_file), str(output_file)], check=True)
        return

    # Create a temporary file list for ffmpeg
    file_list_path = output_file.parent / "file_list.txt"
    with open(file_list_path, "w") as file_list:
        for file in input_files:
            file_list.write(f"file '{file}'\n")

    # Use ffmpeg to concatenate the videos
    command = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(
            file_list_path), "-c", "copy", str(output_file)
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully fused videos into: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error while fusing videos: {e}")
    finally:
        # Clean up the temporary file list
        if file_list_path.exists():
            file_list_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Process subdirectories containing specific .mp4 files and fuse them together.")
    parser.add_argument("root_dir", type=str,
                        help="Root directory to search for subdirectories.")

    args = parser.parse_args()

    try:
        process_video_subdirectories(args.root_dir)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
