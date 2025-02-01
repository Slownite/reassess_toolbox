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
import argparse
from collections import Counter


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


class NpyEdf(Dataset):
    def __init__(self, npy_file, edf_file, schema_file, frame_rate=25, size_image_block=64, window_size=5) -> None:
        self.npy = np.load(npy_file)
        raw = mne.io.read_raw_edf(
            edf_file, preload=False, verbose=False, encoding='latin1')
        annotations = raw.annotations
        # Onset times in seconds
        onset_times = np.round(annotations.onset *
                               frame_rate).astype(int).tolist()
        labels = annotations.description  # Labels for each event
        with open(schema_file, 'r') as file:
            schema = json.load(file)
        self.annotations = {
            k: schema.get(v, 0) for k, v in zip(onset_times, labels)
        }
        self.size = size_image_block
        self.window_size = window_size

        # Expand annotations to include the window around each onset
        self.expanded_annotations = self._expand_annotations()

    def _expand_annotations(self):
        """Expand the annotations to include the window around each onset."""
        expanded = set()
        for onset in self.annotations.keys():
            for offset in range(-self.window_size, self.window_size + 1):
                expanded.add(onset + offset)
        return expanded

    def __len__(self):
        return len(self.npy)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        start = index * self.size
        end = start + self.size

        # Check if any frame in the block falls within the expanded annotation range
        annotation = 1 if any(key for key in range(
            start, end) if key in self.expanded_annotations) else 0
        return self.npy[index], annotation


class MultiNpyEdf(Dataset):
    def __init__(self, npy_files, edf_files, schema_file, frame_rate=25, size_image_block=64, window_size=5) -> None:
        self.data = [NpyEdf(npy_file, edf_file, schema_file, frame_rate, size_image_block, window_size)
                     for npy_file, edf_file in zip(npy_files, edf_files)]
        # Precompute lengths for easier indexing
        self.lengths = [len(dataset) for dataset in self.data]
        self.cumulative_lengths = [0] + \
            list(self._cumulative_sum(self.lengths))

    def _cumulative_sum(self, lengths):
        """Helper method to compute cumulative sums."""
        total = 0
        for length in lengths:
            total += length
            yield total

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}")

        # Find the dataset and local index within that dataset
        dataset_idx = self._find_dataset_index(index)
        local_index = index - self.cumulative_lengths[dataset_idx]

        # Access the corresponding dataset and retrieve the item
        return self.data[dataset_idx][local_index]

    def _find_dataset_index(self, global_index):
        """Find the dataset index for a given global index using cumulative lengths."""
        for i, cum_length in enumerate(self.cumulative_lengths[1:]):
            if global_index < cum_length:
                return i
        raise IndexError(
            f"Global index {global_index} not found in any dataset")


class MultiNpyEdfSequence(Dataset):
    def __init__(self, npy_files, edf_files, schema_file, frame_rate=25, size_image_block=64, window_size=5, sequence_length=10, pad_value=0):
        self.sequence_length = sequence_length
        self.pad_value = pad_value

        self.data = [NpyEdf(npy_file, edf_file, schema_file, frame_rate, size_image_block, window_size)
                     for npy_file, edf_file in zip(npy_files, edf_files)]

        # Precompute lengths for easier indexing
        self.lengths = [len(dataset) for dataset in self.data]
        self.cumulative_lengths = [0] + \
            list(self._cumulative_sum(self.lengths))

    def _cumulative_sum(self, lengths):
        """Helper method to compute cumulative sums."""
        total = 0
        for length in lengths:
            total += length
            yield total

    def __len__(self):
        return sum(self.lengths)

    def _find_dataset_index(self, index):
        """Find which dataset the index belongs to."""
        for i, cumulative_length in enumerate(self.cumulative_lengths[1:]):
            if index < cumulative_length:
                return i
        return len(self.cumulative_lengths) - 1

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}")

        dataset_idx = self._find_dataset_index(index)
        local_index = index - self.cumulative_lengths[dataset_idx]

        # Fetch the sequence
        sequence = []
        labels = []
        for i in range(self.sequence_length):
            idx = local_index + i
            if idx < self.lengths[dataset_idx]:  # If within dataset bounds
                sample = self.data[dataset_idx][idx]
                features, label = torch.tensor(sample[0], dtype=torch.float32), torch.tensor(
                    sample[1], dtype=torch.float32)
            else:  # Apply padding
                features = torch.full_like(sequence[0], self.pad_value)
                label = torch.tensor(0.0)

            sequence.append(features)
            labels.append(label.item())  # Store label as a scalar
        # Ensure sequence dimensions are consistent
        sequence = torch.stack(sequence)  # Stack into a single tensor
        assert sequence.shape[1:] == torch.Size(
            [8192]), f"Feature dimension mismatch: {sequence.shape}"

        # Compute final sequence label: 1 if there's any '1' in the sequence labels, else 0
        final_label = torch.tensor(1.0 if sum(
            labels) > 0.0 else 0.0, dtype=torch.float32)
        return sequence, final_label


def debug_multi_edf_npy():

    parser = argparse.ArgumentParser(prog="test NpyEdf dataset")
    parser.add_argument("--npy", type=pathlib.Path, nargs='+',
                        required=True, help="List of .npy files")
    parser.add_argument("--edf", type=pathlib.Path, nargs='+',
                        required=True, help="List of .edf files")
    parser.add_argument("--schema", type=pathlib.Path, required=True)
    parser.add_argument("-f", "--frame_duration", default=1/25, type=float)
    parser.add_argument("-w", "--window_duration", default=3, type=float)
    args = parser.parse_args()
    dataset = MultiNpyEdf(args.npy, args.edf, args.schema)
    for i in range(len(dataset)):
        input = dataset[i]
        print(
            f"label: {input[1]} at position start {i * 64} end {(i * 64) + 64}")


def debug_multi_edf_npy_sequences():
    parser = argparse.ArgumentParser(prog="test NpyEdf dataset sequences")
    parser.add_argument("--npy", type=pathlib.Path, nargs='+',
                        required=True, help="List of .npy files")
    parser.add_argument("--edf", type=pathlib.Path, nargs='+',
                        required=True, help="List of .edf files")
    parser.add_argument("--schema", type=pathlib.Path, required=True)
    parser.add_argument("-f", "--frame_duration", default=1/25, type=float)
    parser.add_argument("-w", "--window_duration", default=3, type=float)
    parser.add_argument("-s", "--sequence_length", default=10,
                        type=int, help="Length of sequences to retrieve")
    args = parser.parse_args()

    dataset = MultiNpyEdfSequence(
        args.npy, args.edf, args.schema, sequence_length=args.sequence_length)

    for i in range(len(dataset)):
        input_sequence = dataset[i]
        print(
            f"Sequence {i}: Labels {[input[1] for input in input_sequence]} at position start {i * 64} end {(i * 64) + (64 * args.sequence_length)}")


if __name__ == "__main__":
    debug_multi_edf_npy_sequences()
