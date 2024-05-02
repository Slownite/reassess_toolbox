#!/usr/bin/env python3

import argparse
import pathlib
import subprocess


def convert_audio(video_file: pathlib.Path, output_file: pathlib.Path) -> None:
    """
    Convert the audio stream from a video file to an MP3 file using ffmpeg.

    Args:
    video_file (pathlib.Path): The path to the source video file.
    output_file (pathlib.Path): The path where the output MP3 file will be saved.
    """
    cmd = ["ffmpeg", "-i", str(video_file), "-q:a", "0", "-map", "a", str(output_file)]
    subprocess.run(cmd, check=True)


def convert_multiple_audio(
    video_files: pathlib.Path, output_directory: pathlib.Path
) -> None:
    """
    Convert audio streams from multiple video files to MP3 format.

    This function searches for all MP4 files within the given directory and its subdirectories,
    and converts each found video file into an MP3 file stored in the specified output directory.

    Args:
    video_files (pathlib.Path): The directory containing video files to convert.
    output_directory (pathlib.Path): The directory where the output MP3 files will be saved.
    """
    for video_file in video_files.glob("**/*.mp4"):
        output_file = output_directory / f"{video_file.stem}.mp3"
        convert_audio(video_file, output_file)


def run(args) -> None:
    """
    Process the command line arguments and execute the appropriate audio conversion function.

    Args:
    args: Command line arguments including source and destination paths.
    """
    if args.src_path.is_dir() and args.dest_path.is_dir():
        convert_multiple_audio(args.src_path, args.dest_path)
        print("conversion done!")

    elif args.src_path.is_file() and args.dest_path.is_file():
        convert_audio(args.src_path, args.dest_path)
        print("conversion done!")
    else:
        raise ValueError("make sure both path are directories or both files")


def main() -> None:
    """
    Main function to handle command line arguments for the video to audio converter script.

    This script converts video files to audio files, supporting both individual file conversion
    and batch conversion in directories.
    """
    parser = argparse.ArgumentParser(
        prog="video to audio converter",
        descripton="create mp3 files from the audio of mp4 file",
        epilog="use the -h flag to know more",
    )
    parser.add_argument("src_path", type=pathlib.Path)
    parser.add_argument("dest_path", type=pathlib.Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
