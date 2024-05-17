import torch
from torch.utils.data import Dataset
import pathlib
from utils import EEGVideoSynchronizer, VideoStreamer, videos_frame_to_flow


class TwoStreamDatasetRGB(Dataset):
    def __init__(self, path: pathlib.Path, block=5) -> None:
        of = list(path.glob("**/rgb_*.mp4"))
        eeg = list(path.glob("**/*.edf"))
        self.block = block
        self.data = EEGVideoSynchronizer(
            rgb, of=None, egg=eeg, audio=None, block_size_frames=self.block
        )
        self.size = len(VideoStreamer(*rgb, batch=block))

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        frames, compressed_flows, _, annotations, _ = self.data[index]
        center_rgb_frame = torch.tensor(frames[len(frames) // 2]).permute(3, 0, 1, 2)
        return (center_rgb_frame, annotations)


class TwoStreamDatasetOF(Dataset):
    def __init__(self, path: pathlib.Path, block=5) -> None:
        of = list(path.glob("**/flow_*.mp4"))
        egg = list(path.glob("**/*.edf"))
        self.block = block
        self.data = EEGVideoSynchronizer(
            rgb, of, egg, audio, block_size_frames=self.block
        )
        self.size = len(VideoStreamer(*of, batch=block))

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
        return (center_rgb_frame, stack_flows, annotations)
