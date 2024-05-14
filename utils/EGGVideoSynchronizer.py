import shutil
import subprocess
import numpy as np
import sys
import os
from pyedflib import highlevel

import re
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.AudioStream import AudioStream
from utils.EGGReader import EEGStream
from utils.stream import VideoStreamer


from scripts.optical_flow_converter import process_all_videos, video_to_optical_flow


class EEGVideoSynchronizer:

    def __init__(self, rgb, of, eeg, audio, eeg_sampling_rate, block_size_frames):
        """
        Initializes the EEGVideoSynchronizer object.

        Args:
            video_rgb_paths (list[str]): List of paths to video files.
            eeg_path (str): Path to the EEG file.
            video_frame_rate (float): Frame rate of the video in frames per second.
            eeg_sampling_rate (float): Sampling rate of the EEG data in samples per second.
            block_size_frames (int): Number of video frames per block.
            compute_method (str): Method for computing optical flow.
        """
        self.video_frame_rate = 25
        self.eeg_sampling_rate = eeg_sampling_rate
        self.block_size_frames = block_size_frames

        # Initialize the video streamer with all video files found in the directory
        video_rgb_paths = [str(p) for p in rgb]
        self.video_rgb = VideoStreamer(*video_rgb_paths)

        eeg_paths = tuple(str(p) for p in eeg)
        self.eeg_stream = EEGStream(
            *eeg_paths, chunk_duration=self.block_size_frames / video_frame_rate
        )

        video_flow_paths = [str(p) for p in of]
        self.video_flow = VideoStreamer(*video_flow_paths)

        audio_paths = [str(p) for p in audio]
        self.audio_stream = AudioStream(*audio_paths)

        # # Setup paths for concatenated video and EEG files
        # self.video_path = self.find_files(video_directory, "*.mp4")
        # self.eeg_path = self.find_files(eeg_directory, "*.edf")

        # # Initialize readers and capture objects
        # self.video_capture = VideoStreamer(self.video_path)
        # self.eeg_reader = EEGReader(self.eeg_path)

    def find_files(self, directory, pattern):
        """Find all files in a directory that match a certain pattern."""
        return list(Path(directory).glob(pattern))

    def get_item(self, block_idx):
        """
        Gets synchronized video frames, EEG data, and annotations for the specified block index.

        Args:
            block_idx (int): Index of the block to retrieve.

        Returns:
            dict: A dictionary containing synchronized video frames, EEG data, and annotations.
        """
        start_frame = block_idx * self.block_size_frames
        end_frame = start_frame + self.block_size_frames

        # Fetch video frames for each video streamer
        # video_frames = self.video_capture.get_frames(start_frame, end_frame)

        video_frames = self.video_rgb[start_frame:end_frame]
        video_flows = self.video_flow[start_frame:end_frame]

        # Calculate corresponding EEG sample range
        # start_sample = int(start_frame * self.samples_per_frame)
        # end_sample = int(end_frame * self.samples_per_frame)

        eeg_data = self.eeg_stream[block_idx]

        annotations = self.eeg_stream.get_annotations_in_chunk(block_idx)
        audio_data = self.audio_stream[start_frame:end_frame]

        return (video_frames, video_flows, eeg_data, annotations, audio_data)

    def __getitem__(self, value):
        """
        Get a sample of samples of the data stream.

        Args:
            value: The index or slice of indices to retrieve.

        Returns:
            np.ndarray: The frame or frames from the video stream.
        """
        if isinstance(value, slice):
            start, stop, step = value.indices(len(self))
            return np.array([self.get_item(i) for i in range(start, stop, step)])
        elif isinstance(value, int):
            if value >= len(self) or value < 0:
                raise IndexError("Index out of bounds")
            return self.get_item(value)
        else:
            raise TypeError("Invalid argument type")


if __name__ == "__main__":
    directory_path = "C:/Users/hp/OneDrive - Institut National de Statistique et d'Economie Appliquee/Bureau/REASSEASS/data/"
    synchronizer = EEGVideoSynchronizer(
        directory_path, video_frame_rate=20, eeg_sampling_rate=256, block_size_frames=10
    )

    block_data = synchronizer.get_item(0)
    print(block_data)

    video_frames, video_flows, eeg_data, annotations, audio_data = block_data

    print("Number of video frames:", len(video_frames))
    print("EEG data shape:", eeg_data.shape)
    print("Number of annotations:", len(annotations))
    print("Audio data samples:", audio_data.shape)
