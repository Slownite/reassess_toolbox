import cv2
import numpy as np
import torch
import pathlib
import tempfile
import shutil
import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor


def calc_optical_flow_tvl1(prev_gray, curr_gray, optical_flow):
    """Compute TV-L1 optical flow with preallocated flow object."""
    return optical_flow.calc(prev_gray, curr_gray, None)


def save_flow_images(flow, frame_idx, output_dir):
    """Convert optical flow vectors to a color-coded representation and save as an image."""
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Max saturation
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
    # Value represents magnitude
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpeg"), rgb)


def frames_to_video(frames_dir, output_video_path, frame_rate=25):
    """Convert flow images to a video using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-framerate", str(frame_rate),
        "-i", str(frames_dir / "frame_%04d.jpeg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_video_path)
    ]
    subprocess.run(cmd, check=True)


def video_to_optical_flow(video_file, dest_path, compute_method="tvl1", output_format="mp4"):
    """Process a video to compute optical flow and save output."""
    use_temp_dir = output_format == "mp4"
    output_dir = pathlib.Path(
        tempfile.mkdtemp()) if use_temp_dir else dest_path
    output_dir.mkdir(parents=True, exist_ok=True)

    video_cap = cv2.VideoCapture(str(video_file))
    success, prev_frame = video_cap.read()
    if not success:
        print("Failed to read the first frame.")
        return

    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create the TV-L1 Optical Flow object once
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    frame_idx = 0
    futures = []
    # Use multiple threads for saving
    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            success, curr_frame = video_cap.read()
            if not success:
                break  # No more frames

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Compute optical flow
            flow = calc_optical_flow_tvl1(prev_gray, curr_gray, optical_flow)

            # Save the flow image asynchronously
            futures.append(executor.submit(
                save_flow_images, flow, frame_idx, output_dir))

            prev_gray = curr_gray  # Reuse grayscale frames to save memory
            frame_idx += 1

    video_cap.release()
    print("Finished processing all frames.")

    # Wait for all images to be saved
    for future in futures:
        future.result()

    if use_temp_dir:
        frames_to_video(output_dir, dest_path.parent /
                        f"{video_file.stem}.mp4", 25)
        shutil.rmtree(output_dir)


def process_all_videos(source_directory, output_directory, compute_method="tvl1", output_format="mp4"):
    """Process all videos in the source directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for video_file in pathlib.Path(source_directory).glob("**/*.mp4"):
        video_output_path = output_directory / \
            f"{video_file.stem}.{output_format}"
        video_to_optical_flow(video_file, video_output_path,
                              compute_method, output_format)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Video to Optical Flow")
    parser.add_argument("directory", type=pathlib.Path,
                        help="Directory containing video files")
    parser.add_argument("out", type=pathlib.Path, help="Output directory")
    parser.add_argument("-m", "--compute_method", choices=["farneback", "tvl1", "RAFT"], default="tvl1",
                        help="Method to use for optical flow computation")
    parser.add_argument("-o", "--output_format", choices=["jpeg", "mp4"], default="mp4",
                        help="Output format of the results")
    args = parser.parse_args()

    process_all_videos(args.directory, args.out,
                       args.compute_method, args.output_format)


if __name__ == "__main__":
    main()
