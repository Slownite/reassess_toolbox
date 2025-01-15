from pathlib import Path
import subprocess
import argparse


def process_video_subdirectories(root_dir: str, script_path: str):
    """
    Process subdirectories containing 'rgb_*.mp4' or 'flow_*.mp4' files.

    Parameters:
    - root_dir (str): The root directory to search for subdirectories.
    - script_path (str): Path to the script that processes .mp4 files.
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
                run_script(script_path, output_file, rgb_files)

            if flow_files:
                output_file = sub_dir / f"flow_{sub_dir.name}.mp4"
                run_script(script_path, output_file, flow_files)


def run_script(script_path: str, output_file: Path, input_files: list):
    """
    Run the provided script with the given arguments.

    Parameters:
    - script_path (str): Path to the script that processes .mp4 files.
    - output_file (Path): The output .mp4 file.
    - input_files (list): List of input .mp4 files.
    """
    command = ["python", script_path, str(
        output_file)] + [str(file) for file in input_files]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully processed videos into: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error while processing videos: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process subdirectories containing specific .mp4 files.")
    parser.add_argument("root_dir", type=str,
                        help="Root directory to search for subdirectories.")
    parser.add_argument("script_path", type=str,
                        help="Path to the script that processes .mp4 files.")

    args = parser.parse_args()

    try:
        process_video_subdirectories(args.root_dir, args.script_path)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
