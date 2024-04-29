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
import pathlib
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

    # Resize image to make dimensions divisibsudo smartctl -a /dev/sdxle by 8
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


def calculate_optical_flow(prev_frame, curr_frame, frame_idx, compute_method):
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
        raise ValueError(f"Unsupported optical flow method: {compute_method}")

    return flow


def frames_to_video(frames_dir, output_video_path, frame_rate):
    parent = output_video_path.parent
    name = output_video_path.name
    cmd = [
        "ffmpeg",
        "-framerate",
        str(frame_rate),
        "-i",
        str(frames_dir / "frame_%04d.jpeg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(parent / f"flow_{name}"),
    ]
    subprocess.run(cmd, check=True)


def video_to_optical_flow(
    video_file, dest_path, compute_method="farneback", output_format="mp4"
):
    use_temp_dir = output_format == "mp4"
    output_dir = pathlib.Path(tempfile.mkdtemp()) if use_temp_dir else dest_path

    # If the input_data is a path to a video file
    # video_cap = cv2.VideoCapture(str(video_file))
    # success, prev_frame = video_cap.read()
    # if not success:
    #     print("Failed to read the first frame.")
    #     return

    # frame_idx = 0
    # while success:
    #     success, curr_frame = video_cap.read()
    #     if not success:
    #         print("No more frames to process.")
    #         break

    #     print(f"Processing frame {frame_idx}")
    #     flow = calculate_optical_flow(prev_frame, curr_frame, frame_idx, compute_method)
    #     save_flow_images(flow, frame_idx, output_dir)
    #     prev_frame = curr_frame
    #     frame_idx += 1
    # video_cap.release()
    # print("Finished processing all frames.")
    if use_temp_dir:
        # Convert saved images to a video
        print(dest_path / f"flow_{video_file.stem[3:]}.mp4")
        frames_to_video(output_dir, dest_path / video_file.stem / ".mp4", 25)
        shutil.rmtree(output_dir)


def process_all_videos(
    source_directory, output_directory, compute_method, output_format
):
    """Process all video files in the specified directory, storing results in a separate output directory."""
    # Create the output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    # Process each video file in the source directory
    for video_file in pathlib.Path(source_directory).glob(
        "**/*.mp4"
    ):  # Adjust glob pattern if other video formats are needed
        video_output_path = output_directory / video_file.stem
        video_to_optical_flow(
            video_file, video_output_path, compute_method, output_format
        )


def main():
    parser = argparse.ArgumentParser(description="Convert Video to Optical Flow")
    parser.add_argument(
        "directory", type=pathlib.Path, help="Directory containing video files"
    )
    parser.add_argument("out", type=pathlib.Path, help="output directory")
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
