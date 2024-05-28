#!/usr/bin/env python3
import torch
from ..modules import I3D
from torch.utils.data import DataLoader, Dataset
from torch import nn
from ..utils import VideoStreamer
class I3DDatasetRGB(Dataset):
    def __init__(
        self, path: pathlib.Path, block=66
    ) -> None:
        rgb = list(path.glob("**/rgb_*.mp4"))
        self.block = block
        self.data = VideoStreamer(*map(str, rgb), batch=block)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        rgb_frames = self.data[index]
        assert rgb_frames.shape == (
            self.block,
            224,
            224,
        ), f"rgb_frames shape is {rgb_frames.shape}, should be ({self.block}, 224, 224) "
        return rgb_frames

class I3DDatasetOF(Dataset):
    def __init__(
        self, path: pathlib.Path, block=66
    ) -> None:
        of = list(path.glob("**/flow_*.mp4"))
        self.block = block
        self.data = VideoStreamer(*map(str, flow), batch=block)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        compressed_flows = self.data[index]
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
        return stack_flows
def init(args)->[I3D, DataLoader]:
    if args.model = "i3d_rgb":
        model = I3D(2)

def main():
    pass
