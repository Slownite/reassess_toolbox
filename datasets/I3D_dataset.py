#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import pathlib
from ..utils import (
    EEGVideoSynchronizer,
    VideoStreamer,
    videos_frame_to_flow,
    one_hot_encoding,
)
import json


class I3DDatasetRGB(Dataset):
    def __init__(
        self, path: pathlib.Path, annotation_schema_path: pathlib.Path, block=75
    ) -> None:
        rgb = list(path.glob("**/rgb_*.mp4"))
        eeg = list(path.glob("**/*.edf"))
        self.block = block
        self.data = EEGVideoSynchronizer(rgb=rgb, eeg=eeg, block_size_frames=self.block)
        self.size = len(VideoStreamer(*map(str, rgb), batch=block))
        with open(annotation_schema_path, "r") as file:
            self.annotation_schema = json.load(file)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        rgb_frames, _, _, annotations, _ = self.data[index]
        assert rgb_frames.shape == (
            self.block,
            224,
            224,
        ), f"rgb_frames shape is {rgb_frames.shape}, should be ({self.block}, 224, 224) "
        numerical_annotations = one_hot_encoding(self.annotation_schema, annotations)
        return (rgb_frames, annotations)


class I3DDatasetOF(Dataset):
    def __init__(
        self, path: pathlib.Path, annotation_schema_path: pathlib.Path, block=75
    ) -> None:
        of = list(path.glob("**/flow_*.mp4"))
        eeg = list(path.glob("**/*.edf"))
        self.block = block
        self.data = EEGVideoSynchronizer(of=of, eeg=eeg, block_size_frames=self.block)
        self.size = len(VideoStreamer(*map(str, of), batch=block))
        with open(annotation_schema_path, "r") as file:
            self.annotation_schema = json.load(file)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        _, compressed_flows, _, annotations, _ = self.data[index]
        assert compressed_flows.shape == (
            self.block,
            224,
            224,
        ), f"compressed_flows shape is {compressed_flow.shape}, should be ({self.block}, 224, 224)"
        uncompressed_flow = torch.tensor(videos_frame_to_flow(flows)).permute(
            3, 0, 1, 2
        )
        uf_vectors, uf_depth, uf_width, uf_height = uncompressed_flow.shape
        stack_flows = uncompressed_flow.reshape(
            uf_depth * uf_vectors, uf_width, uf_width
        )
        numerical_annotations = one_hot_encoding(self.annotation_schema, annotations)
        return (stack_flows, numerical_annotations)
