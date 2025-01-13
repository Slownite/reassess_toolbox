from pathlib import Path
import subprocess
import argparse


def concatenate_videos(output_file: str, input_files: list):
    """
    Concatenate multiple video files into a single video using ffmpeg.

    Parameters:
    - output_file (str): Path to the output video file.
    - input_files (list): List of paths to input video files.
    """
    # Convert input files to Path objects
    input_paths = [Path(file) for file in input_files]

    # Check if all input files exist
    for file in input_paths:
        if not file.exists():
            raise FileNotFoundError(f"Input file not found: {file}")

    # Create a temporary file for the file list
    temp_file = Path("file_list.txt")
    with temp_file.open("w") as f:
        for file in input_paths:
            # ffmpeg requires the file paths to be prefixed with "file "
            f.write(f"file '{file.as_posix()}'\n")

    # Run ffmpeg to concatenate the videos
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", temp_file.as_posix(),
                "-c", "copy",
                output_file
            ],
            check=True
        )
        print(f"Concatenation complete. Output file: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during concatenation: {e}")
    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple videos into one using ffmpeg.")
    parser.add_argument("output_file", type=str,
                        help="Path to the output video file.")
    parser.add_argument("input_files", type=str, nargs='+',
                        help="Paths to input video files.")

    args = parser.parse_args()

    try:
        concatenate_videos(args.output_file, args.input_files)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
