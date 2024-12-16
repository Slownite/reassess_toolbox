from ultralytics import YOLO
from argparse import ArgumentParser
import os
import pathlib
import cv2 as cv


def read_first_n_frames(video_path: str, n: int = 1):
    """
    Reads the first N frames from a video file.

    Args:
        video_path (str): The path to the video file.
        n (int): The number of frames to read.

    Returns:
        list: A list of frames (each frame is a NumPy array) from the video.
    """
    frames = []
    cap = cv.VideoCapture(str(video_path))  # Open the video file

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return frames

    frame_count = 0

    while frame_count < n:
        ret, frame = cap.read()

        if not ret:
            print(f"Reached the end of the video after {frame_count} frames.")
            break

        frames.append(frame)
        frame_count += 1

    cap.release()
    print(f"Successfully read {len(frames)} frames from {video_path}.")

    return frames


def cropped(boxes, frame):

    smallest_person = min(
        boxes,
        key=lambda box: (box.xyxy[0][2] - box.xyxy[0]
                         [0]) * (box.xyxy[0][3] - box.xyxy[0][1])
    )
    x_min, y_min, x_max, y_max = map(
        int, smallest_person.xyxy[0])  # Convert to integers

    # Ensure the bounding box stays within frame boundaries
    height, width, _ = frame.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    # Crop the frame to focus on the detected person
    return x_max - x_min, y_max - y_min, x_min, y_min


def run_file(filename: pathlib.Path, output_file: pathlib.Path, weights: str, conf=0.5, iou=0.1):
    frame = read_first_n_frames(filename)[0]
    yolo = YOLO(weights)
    result = yolo.predict(frame, conf=conf, iou=iou)
    boxes = result[0].boxes
    width, height, x_min, y_min = cropped(boxes, frame)
    ffmpeg_command = (
        f"ffmpeg -y -i {filename} "
        f"-vf \"crop={width}:{height}:{x_min}:{y_min}\" "
        f"-c:v libx264 -crf 23 -preset fast {output_file}"
    )
    os.system(ffmpeg_command)


def main():
    parser = ArgumentParser()
    parser.add_argument("src", type=pathlib.Path)
    parser.add_argument("dest", type=pathlib.Path)
    parser.add_argument("-w", "--weights", type=str, default="yolo11x")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.1)
    args = parser.parse_args()
    if args.src.is_file():
        run_file(args.src, args.dest, args.weights, args.conf, args.iou)
    else:
        for video in args.src.glob("*.mp4"):
            run_file(video, args.dest / video.name,
                     args.weights, args.conf, args.iou)


if __name__ == "__main__":
    main()
