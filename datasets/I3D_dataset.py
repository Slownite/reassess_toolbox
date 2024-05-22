#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import pathlib
from utils import (
    EEGVideoSynchronizer,
    VideoStreamer,
    videos_frame_to_flow,
    one_hot_encoding,
)


class I3DDatasetRGB(Dataset):
    def __init__(
        self, path: pathlib.Path, annotation_schema_path: pathlib.Path, block=75
    ) -> None:
        rgb = list(path.glob("**/rgb_*.mp4"))
        egg = list(path.glob("**/*.edf"))
        self.block = block
        self.data = EEGVideoSynchronizer(
            rgb, of, egg, audio, block_size_frames=self.block
        )
        self.size = len(VideoStreamer(*of, batch=block))
        with open(annotation_schema_path, "r") as file:
            self.annotation_scheme = json.load(file)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        rgb_frames, _, _, annotations, _ = self.data[index]
        numerical_annotations = np.stack(
            [one_hot_encoding(self.annotation_scheme, i) for i in annotations]
        )
        return (rgb_frames, annotations)


class I3DDatasetOF(Dataset):
    def __init__(
        self, path: pathlib.Path, annotation_schema_path: pathlib.Path, block=75
    ) -> None:
        of = list(path.glob("**/flow_*.mp4"))
        egg = list(path.glob("**/*.edf"))
        self.block = block
        self.data = EEGVideoSynchronizer(
            rgb, of, egg, audio, block_size_frames=self.block
        )
        self.size = len(VideoStreamer(*of, batch=block))
        with open(annotation_schema_path, "r") as file:
            self.annotation_scheme = json.load(file)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        _, compressed_flows, _, annotations, _ = self.data[index]
        uncompressed_flow = torch.tensor(videos_frame_to_flow(flows)).permute(
            3, 0, 1, 2
        )
        uf_vectors, uf_depth, uf_width, uf_height = uncompressed_flow.shape
        stack_flows = uncompressed_flow.reshape(
            uf_depth * uf_vectors, uf_width, uf_width
        )
        numerical_annotations = np.stack(
            [one_hot_encoding(self.annotation_scheme, i) for i in annotations]
        )
        return (stack_flows, numerical_annotations)
