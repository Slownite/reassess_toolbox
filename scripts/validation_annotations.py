import argparse
from pathlib import Path
import cv2


def count_video_frames(video_path):
    """
    Count the number of frames in a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        int: Number of frames in the video.
    """
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def count_annotation_lines(annotation_path):
    """
    Count the number of lines in an annotation file.

    Args:
        annotation_path (Path): Path to the annotation file.

    Returns:
        int: Number of lines in the annotation file.
    """
    with open(annotation_path, 'r') as file:
        return sum(1 for _ in file)


def validate_annotations(directory):
    """
    Validate that the sum of frames in all videos matches the lines in the annotation file.

    Args:
        directory (str): Path to the directory containing videos and one annotation file.
    """
    directory_path = Path(directory)

    # Find video files
    # Adjust extension if needed
    video_files = list(directory_path.glob("*.mp4"))
    annotation_files = list(directory_path.glob("*.txt"))

    if not video_files:
        print(f"No video files found in directory '{directory}'.")
        return

    if len(annotation_files) != 1:
        print(f"Error: Expected exactly one annotation file in directory '{
              directory}', but found {len(annotation_files)}.")
        return

    annotation_file = annotation_files[0]

    # Calculate total frames in all videos
    total_frames = 0
    for video in video_files:
        try:
            frame_count = count_video_frames(video)
            total_frames += frame_count
        except Exception as e:
            print(f"Error processing video {video.name}: {e}")
            return

    # Count lines in the annotation file
    try:
        annotation_count = count_annotation_lines(annotation_file)
    except Exception as e:
        print(f"Error reading annotation file {annotation_file.name}: {e}")
        return

    # Validate total frames vs annotation lines
    if total_frames == annotation_count:
        print(f"Valid: Total frames ({
              total_frames}) matches annotation lines ({annotation_count}).")
    else:
        print(f"Mismatch: Total frames ({
              total_frames}) does not match annotation lines ({annotation_count}).")


def main():
    parser = argparse.ArgumentParser(
        description="Validate that the total video frames match annotation lines.")
    parser.add_argument("directory", type=str,
                        help="Path to the directory containing videos and annotations.")
    args = parser.parse_args()

    validate_annotations(args.directory)


if __name__ == "__main__":
    main()
