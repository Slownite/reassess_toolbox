import shutil
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
from tempfile import TemporaryDirectory

##### Funtion Using the method Farneback (Desnse optical flow) #####

def calc_optical_flow_farneback(prev_frame, curr_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
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
    print("Loaaaaaad model")
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
    flow_up = output['flow'] if isinstance(output, dict) and 'flow' in output else output[1]

    # Convert flow to numpy array for further processing
    flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
    return flow_up


# image1 and image2 should be loaded using torchvision.io.read_image ,
# so they are in the shape of CxHxW and dtype torch.uint8


def save_flow_images(flow, frame_idx, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.png"), rgb)
    print(f"Saved frame {frame_idx}")

def video_to_optical_flow(src_path, dest_path, compute_method="RAFT", output_format="mp4"):
    os.makedirs(dest_path, exist_ok=True)  # Ensure destination directory exists
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
            prev_frame_tensor = torch.from_numpy(prev_frame).permute(2, 0, 1).float() / 255.0
            curr_frame_tensor = torch.from_numpy(curr_frame).permute(2, 0, 1).float() / 255.0
            flow = calc_optical_flow_raft(preprocess_image_for_raft(prev_frame_tensor),
                                          preprocess_image_for_raft(curr_frame_tensor))
        elif compute_method == "farneback":
            flow = calc_optical_flow_farneback(prev_frame, curr_frame)
        elif compute_method == "tvl1":
            flow = calc_optical_flow_tvl1(prev_frame, curr_frame)
        else:
            print("Unsupported method")
            break

        save_flow_images(flow, frame_idx, dest_path)
        prev_frame = curr_frame
        frame_idx += 1

    video_cap.release()

    if output_format == "mp4":
        frames_to_video(dest_path, os.path.join(dest_path, "output_video.mp4"))

    print("Finished processing all frames.")
            
def frames_to_video(frames_dir, output_video_path, frame_rate=30):
   
    frames_path = str(Path(frames_dir) / "frame_%04d.png")
    cmd = [
        "ffmpeg", "-framerate", str(frame_rate), "-i", frames_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video_path
    ]
    print("Running FFmpeg command:", ' '.join(cmd))  # Print the command to check it
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Convert Video to Optical Flow")
    parser.add_argument("src_path", help="Source video path")
    parser.add_argument("dest_path", help="Destination path for optical flow video or images")
    parser.add_argument("-m", "--compute_method", choices=["farneback", "tvl1", "RAFT"], help="Method to use for optical flow computation")
    parser.add_argument("-o", "--output_format", choices=["png", "mp4"], default="mp4",help="Output format of the results")

    args = parser.parse_args()

    video_to_optical_flow(args.src_path, args.dest_path, args.compute_method, args.output_format)
if __name__ == "__main__":
    video_path = "C:/Users/hp/OneDrive - Institut National de Statistique et d'Economie Appliquee/Bureau/REASSEASS/data/video1.ASF"
    output_path = r"C:\Users\hp\OneDrive - Institut National de Statistique et d'Economie Appliquee\Bureau\REASSEASS\RAFTtest"
    video_to_optical_flow(video_path, output_path, "RAFT","png")