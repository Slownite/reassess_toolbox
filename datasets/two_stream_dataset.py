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


class TwoStreamDatasetRGB(Dataset):
    def __init__(
        self, path: pathlib.Path, annotation_schema_path: pathlib.Path, block=75
    ) -> None:
        of = list(path.glob("**/rgb_*.mp4"))
        eeg = list(path.glob("**/*.edf"))
        self.annotation_scheme
        self.block = block
        self.data = EEGVideoSynchronizer(
            rgb, of=None, egg=eeg, audio=None, block_size_frames=self.block
        )
        self.size = len(VideoStreamer(*rgb, batch=block))
        with open(annotation_schema_path, "r") as file:
            self.annotation_schema = json.load(file)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        frames, _, _, annotations, _ = self.data[index]
        assert frames.shape == (
            self.block,
            224,
            224,
        ), f"rgb_frames shape is {frames.shape}, should be ({self.block}, 224, 224) "
        center_rgb_frame = torch.tensor(frames[len(frames) // 2]).permute(3, 0, 1, 2)
        numerical_annotations = one_hot_encoding(self.annotation_schema, annotations)
        return (center_rgb_frame, numerical_annotations)


class TwoStreamDatasetOF(Dataset):
    def __init__(self, path: pathlib.Path, block=75) -> None:
        of = list(path.glob("**/flow_*.mp4"))
        egg = list(path.glob("**/*.edf"))
        self.block = block
        self.data = EEGVideoSynchronizer(
            rgb, of, egg, audio, block_size_frames=self.block
        )
        self.size = len(VideoStreamer(*of, batch=block))
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
