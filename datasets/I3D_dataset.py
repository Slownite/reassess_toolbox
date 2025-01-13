#!/usr/bin/env python3
from collections import Counter
import torch
from torch.utils.data import Dataset
import pathlib
import json
from utils import (
    process_annotation_text_file,
    VideoStreamer,
    videos_frame_to_flow
)
from more_itertools import flatten
import numpy as np
import mne


def standardize(rgb, flow):
    if rgb.shape[0] > flow.shape[0]:
        new_rgb = rgb[:len(flow)]
        return new_rgb, flow
    elif rgb.shape[0] < flow.shape[0]:
        new_flow = flow[:len(rgb)]
        return rgb, new_flow
    else:
        return rgb, flow


class I3D_embeddings(Dataset):
    def __init__(
            self, path: pathlib.Path, annotation_schema_path: pathlib.Path, policy: str = "two_class_policy",
    ) -> None:
        policies = {}
        policies['two_class_policy'] = lambda x: 1 if 1 in x else 0
        with open(annotation_schema_path, 'r') as file:
            self.schema = json.load(file)

        all_files = sorted(path.glob("**/*"))
        rgb_npy_files = [f for f in all_files if f.suffix ==
                         '.npy' and f.name.startswith("rgb_")]
        flow_npy_files = [f for f in all_files if f.suffix ==
                          '.npy' and f.name.startswith("flow_")]
        txt_files = [f for f in all_files if f.suffix == '.txt']
        self.annotations = list(flatten([process_annotation_text_file(
            path, self.schema, policies[policy]) for path in txt_files]))
        rgb_list = [np.load(npy_path) for npy_path in rgb_npy_files]
        flow_list = [np.load(npy_path) for npy_path in flow_npy_files]
        rgb_tensors = torch.tensor(np.concatenate(rgb_list, axis=0)).squeeze()
        flow_tensors = torch.tensor(
            np.concatenate(flow_list, axis=0)).squeeze()
        self.rgb_tensors, self.flow_tensors = standardize(
            rgb_tensors, flow_tensors)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        return (self.rgb_tensors[index], self.flow_tensors[index]), self.annotations[index]


class NpyEdfDataset(Dataset):
    def __init__(self, file_paths, edf_paths, annotation_json, frame_duration=0.01, window_duration=1.0):
        """
        Args:
            file_paths (list of str): List of paths to the .npy files.
            edf_paths (list of str): List of paths to the .edf files (same order as .npy files).
            annotation_json (str): Path to a JSON file mapping annotations to replacements.
            frame_duration (float): Duration of each frame in seconds (e.g., 0.01 for 10ms).
            window_duration (float): Duration of the time window around an event to be marked as 1 (in seconds).
        """
        assert len(file_paths) == len(
            edf_paths), "Number of .npy and .edf files must match."

        self.file_paths = file_paths
        self.edf_paths = edf_paths
        self.frame_duration = frame_duration
        self.window_duration = window_duration

        # Load the annotation replacement map from JSON
        with open(annotation_json, 'r') as f:
            self.annotation_map = json.load(f)

        self.current_file_index = 0
        self.current_position_in_file = 0
        self.file_data = None
        self.annotations = None

        # Load the first file and its annotations
        self._load_file(
            self.file_paths[self.current_file_index], self.edf_paths[self.current_file_index])

        # Create a mapping of file index to data indices for len and indexing
        self.file_start_indices = [0]
        self.total_length = 0

        for file_path in self.file_paths:
            data = np.load(file_path)
            self.total_length += len(data)
            self.file_start_indices.append(self.total_length)

    def _load_file(self, file_path, edf_path):
        """Load data from the given .npy file and its corresponding .edf file."""
        self.file_data = np.load(file_path)
        self.annotations = self._load_edf_annotations(edf_path)

    def _load_edf_annotations(self, edf_path):
        """Load annotations from an .edf file using MNE."""
        raw = mne.io.read_raw_edf(
            edf_path, preload=False, verbose=False, encoding='latin1')
        annotations = raw.annotations

        onset_times = annotations.onset  # Onset times in seconds
        durations = annotations.duration  # Duration of each event
        labels = annotations.description  # Labels for each event

        # Convert onset times to embedding indices
        embedding_indices = [int(onset / (self.frame_duration * 64))
                             for onset in onset_times]

        return list(zip(embedding_indices, durations, labels))

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        """
        Retrieve an item by its global index.

        Args:
            index (int): Global index across all files.

        Returns:
            tuple: (data, annotations) for the corresponding embedding.
        """
        if index < 0 or index >= self.total_length:
            raise IndexError("Index out of range")

        # Determine which file this index belongs to
        for file_idx in range(len(self.file_start_indices) - 1):
            start = self.file_start_indices[file_idx]
            end = self.file_start_indices[file_idx + 1]
            if start <= index < end:
                if file_idx != self.current_file_index:
                    self.current_file_index = file_idx
                    self._load_file(
                        self.file_paths[self.current_file_index], self.edf_paths[self.current_file_index])

                self.current_position_in_file = index - start
                break

        data = self.file_data[self.current_position_in_file]
        annotation = self._get_annotation_for_index(
            self.current_position_in_file)

        return data, annotation

    def _get_annotation_for_index(self, index):
        """
        Retrieve the annotation for a given embedding index, including a time window around events.

        Args:
            index (int): Index of the embedding within the current file.

        Returns:
            int: 1 if the annotation or its surrounding window is mapped to a value in the JSON, 0 otherwise.
        """
        window_frames = int(self.window_duration / (self.frame_duration * 64))

        for embedding_index, duration, label in self.annotations:
            if abs(embedding_index - index) <= window_frames:
                return self.annotation_map.get(label, 0)

        return 0

    def get_current_file_info(self):
        """
        Get the current file and position within that file.

        Returns:
            tuple: (current_file_path, current_position_in_file)
        """
        return self.file_paths[self.current_file_index], self.current_position_in_file


# Example usage:
if __name__ == "__main__":
    # List of .npy and .edf files
    file_paths = ["file1.npy", "file2.npy", "file3.npy"]
    edf_paths = ["file1.edf", "file2.edf", "file3.edf"]
    annotation_json = "annotations.json"  # Path to the annotation mapping JSON

    # Create the dataset
    dataset = NpyEdfDataset(file_paths, edf_paths, annotation_json)

    # Access data
    for i in range(len(dataset)):
        data, annotation = dataset[i]
        current_file, position = dataset.get_current_file_info()
        print(f"Data: {data}, Annotation: {annotation}, File: {
              current_file}, Position: {position}")
