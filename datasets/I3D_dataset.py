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


class Embedding(Dataset):
    def __init__(
            self, path: pathlib.Path, annotations_path; pathlib.Path, annotation_schema_path: pathlib.Path
    ) -> None:


    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
