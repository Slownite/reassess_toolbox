import cv2
from cv2 import optflow
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
import math
import argparse
import os
import subprocess
from pathlib import Path
import tempfile
import shutil

##### Funtion Using the method Farneback (Desnse optical flow) #####


def calc_optical_flow_farneback(prev_frame, curr_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    return flow


##### Funtion Using the method TLV1  #####


def calc_optical_flow_tvl1(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Create Optical Flow object
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    flow = optical_flow.calc(prev_gray, curr_gray, None)

    return flow


##### Funtion Using THE RAFT MODEL #####


def preprocess_image_for_raft(image_tensor):
    """
    Preprocess the image tensor for RAFT.
    Converts pixel values to [-1, 1] and ensures the dimensions are divisible by 8.
    """
    print("    Preprocess the image tensor for RAFT.")

    if image_tensor.ndim == 3:
        _, h, w = image_tensor.shape  # Normal case
    elif image_tensor.ndim == 4:
        _, _, h, w = image_tensor.shape  # Batch dimension included

    # Normalize pixel values to [-1, 1]
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Resize image to make dimensions divisible by 8
    new_h = math.ceil(h / 8) * 8
    new_w = math.ceil(w / 8) * 8
    resize = T.Resize((new_h, new_w))

    image_tensor = resize(image_tensor)
    image_tensor = normalize(image_tensor)

    # Ensure batch dimension is present for RAFT processing
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def calc_optical_flow_raft(prev_frame, curr_frame):
    print("Load model")
    model = raft_large(weights=True, progress=True)
    model.eval()

    # Preprocess images
    prev_frame = preprocess_image_for_raft(prev_frame)
    curr_frame = preprocess_image_for_raft(curr_frame)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prev_frame, curr_frame = prev_frame.to(device), curr_frame.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(prev_frame, curr_frame)

    # Check if output is a dictionary and contains 'flow', otherwise assume it's a tuple
    flow_up = (
        output["flow"] if isinstance(output, dict) and "flow" in output else output[1]
    )

    # Convert flow to numpy array for further processing

    flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
    return flow_up


# image1 and image2 should be loaded using torchvision.io.read_image ,
# so they are in the shape of CxHxW and dtype torch.uint8


def save_flow_images(flow, frame_idx, output_dir):
    """
    Converts the optical flow vectors into polar coordinates (magnitude and angle) to represent the flow's intensity and direction.
    Encodes the flow direction as hue and the magnitude as value in an HSV image. The saturation is set to maximum (255) to ensure color intensity.
    Converts the HSV image to RGB for saving as a standard image format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpeg"), rgb)
    print(f"Saved frame {frame_idx}")


def video_to_optical_flow(
    src_path, dest_path, compute_method="farneback", output_format="mp4"
):
    """Process video to compute optical flow, saving results as images or a single video."""
    use_temp_dir = output_format == "mp4"
    if use_temp_dir:
        temp_dir = tempfile.mkdtemp(dir=dest_path)
        output_dir = temp_dir
    else:
        os.makedirs(dest_path, exist_ok=True)
        output_dir = dest_path

    video_cap = cv2.VideoCapture(src_path)
    success, prev_frame = video_cap.read()
    if not success:
        print("Failed to read the first frame.")
        return

    frame_idx = 0
    while success:
        success, curr_frame = video_cap.read()
        if not success:
            print("No more frames to process.")
            break

        print(f"Processing frame {frame_idx}")
        if compute_method == "RAFT":
            prev_frame_tensor = (
                torch.from_numpy(prev_frame).permute(2, 0, 1).float() / 255.0
            )
            curr_frame_tensor = (
                torch.from_numpy(curr_frame).permute(2, 0, 1).float() / 255.0
            )
            flow = calc_optical_flow_raft(
                preprocess_image_for_raft(prev_frame_tensor),
                preprocess_image_for_raft(curr_frame_tensor),
            )
        elif compute_method == "farneback":
            flow = calc_optical_flow_farneback(prev_frame, curr_frame)
        elif compute_method == "tvl1":
            flow = calc_optical_flow_tvl1(prev_frame, curr_frame)
        else:
            print("Unsupported method")
            break

        save_flow_images(flow, frame_idx, output_dir)
        prev_frame = curr_frame
        frame_idx += 1

    video_cap.release()

    if use_temp_dir:
        frames_to_video(temp_dir, os.path.join(dest_path, Path(src_path).stem + ".mp4"))
        shutil.rmtree(
            temp_dir
        )  # Remove the temporary directory after creating the video

    print("Finished processing all frames.")


def frames_to_video(frames_dir, output_video_path, frame_rate):
    parent = output_video_path.parent
    name = output_video_path.name
    frames_path = str(Path(frames_dir) / "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-framerate",
        str(frame_rate),
        "-i",
        frames_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(parent / f"flow_{name}"),
    ]
    subprocess.run(cmd, check=True)


def process_all_videos(
    source_directory, output_directory, compute_method, output_format
):
    """Process all video files in the specified directory, storing results in a separate output directory."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each video file in the source directory
    for video_file in Path(source_directory).glob(
        "**/*.mp4"
    ):  # Adjust glob pattern if other video formats are needed
        video_output_path = os.path.join(
            output_directory, video_file.stem
        )  # Create a subdirectory for each video's output
        os.makedirs(video_output_path, exist_ok=True)
        video_to_optical_flow(
            str(video_file), video_output_path, compute_method, output_format
        )


def main():
    parser = argparse.ArgumentParser(description="Convert Video to Optical Flow")
    parser.add_argument("directory", help="Directory containing video files")
    parser.add_argument("out", help="output directory")
    parser.add_argument(
        "-m",
        "--compute_method",
        choices=["farneback", "tvl1", "RAFT"],
        default="tvl1",
        help="Method to use for optical flow computation",
    )
    parser.add_argument(
        "-o",
        "--output_format",
        choices=["jpeg", "mp4"],
        default="mp4",
        help="Output format of the results",
    )

    args = parser.parse_args()
    process_all_videos(
        args.directory, args.out, args.compute_method, args.output_format
    )


if __name__ == "__main__":
    main()
