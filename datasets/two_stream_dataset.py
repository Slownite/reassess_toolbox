import torch
from torch.utils.data import Dataset
import pathlib
from ..utils import EEGVideoSynchronizer, VideoStreamer


class TwoStreamDataset(Dataset):
    def __init__(self, path: pathlib.Path, block=5) -> None:
        rgb = list(path.glob("**/rgb_*.mp4"))
        of = list(path.glob("**/flow_*.mp4"))
        egg = list(path.glob("**/*.edf"))
        self.data = EEGVideoSynchronizer(rgb, of, egg, block_size_frames=block)
        self.size = len(VideoStreamer(*of, batch=block))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        frames, flows, _, _, annotations = self.data[index]
        return (frames[len(frames) // 2], flows, annotations)
