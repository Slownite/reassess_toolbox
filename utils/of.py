#!/usr/bin/env python3
import numpy as np
import cv2


def rgb_to_flow(rgb_image):
    """
    Convert an RGB-encoded optical flow image back to the flow format (w, h, 2).

    Parameters:
    - rgb_image (numpy.ndarray): Input RGB image encoded from optical flow.

    Returns:
    - numpy.ndarray: An array of shape (w, h, 2) containing the x and y flow components.
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Extract the angle (H channel) and magnitude (V channel)
    h_channel = hsv[..., 0].astype(np.float32)
    v_channel = hsv[..., 2].astype(np.float32)

    # Convert hue angle back to radians
    ang = (h_channel * 2) * np.pi / 180

    # Reverse the normalization of magnitude
    # Assume we know the original max magnitude or estimate it (e.g., using the max V channel value)
    max_mag = np.max(v_channel)
    mag = v_channel * max_mag / 255

    # Convert polar to Cartesian coordinates
    flow_x = mag * np.cos(ang)
    flow_y = mag * np.sin(ang)

    # Combine x and y components into one array
    flow = np.stack((flow_x, flow_y), axis=-1)

    return flow


def videos_frame_to_flow(frames: np.ndarray) -> np.ndarray:
    """
    Converts a batch of RGB optical flow frames to two-channel flow format using NumPy.

    Parameters:
    - rgb_frames (numpy.ndarray): Input array of shape (num, w, h, 3) where:
        - num is the number of frames,
        - w and h are the width and height of the frames respectively.
        - 3 stands for the RGB channels,

    Returns:
    - numpy.ndarray: An array of shape (num, w, h, 2) containing the x and y flow components.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"The shape of the numpy array is {frames.shape} but it should of 4 dimensions and the last one should be 3."
        )
    flow_array = np.array([rgb_to_flow(frame) for frame in frames])
    return flow_array
