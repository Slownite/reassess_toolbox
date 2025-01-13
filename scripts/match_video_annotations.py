import argparse
import csv
from pathlib import Path
import mne
import cv2


def get_video_duration(video_file):
    """
    Get the duration of a video file in seconds.

    Args:
        video_file (str): Path to the video file.

    Returns:
        float: Duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_file}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        raise ValueError(f"Cannot determine FPS for video file: {video_file}")
    return frame_count / fps


def extract_annotations_with_video(edf_file, video_files, output_csv):
    """
    Extract annotations and their timestamps from an EDF file and associate them with video files based on duration.

    Args:
        edf_file (str): Path to the EDF file.
        video_files (list): List of video files in order.
        output_csv (str): Path to the output CSV file.
    """
    try:
        # Load the raw EDF file with Latin-1 encoding
        raw = mne.io.read_raw_edf(
            edf_file, preload=False, verbose=False, encoding="latin1")
        annotations = raw.annotations
        total_edf_duration = raw.times[-1]
    except Exception as e:
        raise ValueError(f"Failed to read EDF file: {e}")

    # Calculate durations of videos
    video_durations = []
    for video_file in video_files:
        video_durations.append(get_video_duration(video_file))

    total_video_duration = sum(video_durations)
    if abs(total_video_duration - total_edf_duration) > 1e-2:
        raise ValueError(
            f"Mismatch between total video duration ({total_video_duration:.2f}s) and EDF duration ({total_edf_duration:.2f}s)."
        )

    # Assign annotations to videos
    extracted_annotations = []
    video_index = 0
    video_time_accumulator = 0

    for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
        # Advance to the correct video based on onset time
        while video_index < len(video_durations) and onset >= video_time_accumulator + video_durations[video_index]:
            video_time_accumulator += video_durations[video_index]
            video_index += 1

        if video_index >= len(video_durations):
            raise ValueError("Annotation onset exceeds total video duration.")

        # Calculate onset relative to the current video
        video_relative_onset = onset - video_time_accumulator
        extracted_annotations.append(
            [video_relative_onset, description, Path(video_files[video_index]).name])

    # Write to CSV
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Onset (s)", "Annotation", "Video File"])
            writer.writerows(extracted_annotations)
        print(f"Annotations successfully saved to {output_csv}")
    except Exception as e:
        raise ValueError(f"Failed to write to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract annotations from an EDF file and associate them with video files based on duration.")
    parser.add_argument("edf_file", type=str, help="Path to the EDF file.")
    parser.add_argument("video_files", nargs='+', type=str,
                        help="Paths to the corresponding video files in order.")
    parser.add_argument("output_csv", type=str,
                        help="Path to the output CSV file.")
    args = parser.parse_args()

    edf_path = Path(args.edf_file)
    video_paths = [Path(video) for video in args.video_files]

    if not edf_path.is_file():
        print(
            f"Error: The EDF file '{args.edf_file}' does not exist or is not a file.")
        return

    for video_path in video_paths:
        if not video_path.is_file():
            print(
                f"Error: The video file '{video_path}' does not exist or is not a file.")
            return

    try:
        extract_annotations_with_video(
            args.edf_file, args.video_files, args.output_csv)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
