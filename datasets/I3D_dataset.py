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
import json
import re
def extract_number(string: str):
        array = string.split("_")[2]
def find_matches(directory=pathlib.Path("."), prefix="rgb_", suffix="Mixed_5c.pt")->dict[pathlib.Path, list[pathlib.Path]]:
        embedding_files = directory.glob(f"{prefix}*.pt")
        annotations_files = directory.glob("*.txt")
        pattern_embedding =
        pattern_edf = re.compile("annotations_patient\\d+-(ev)?(\\d)_edf.txt")
        result = {}
        assert pattern_embedding.match("rgb_Beige24052003-12934503-2_1_Mixed_5c.pt").group(1) == 2, "the result should be 2"
        assert pattern_embedding.match("rgb_Beige24052003-12934503_1_Mixed_5c.pt").group(2) == 1, "the result should be 1"

        # for annotations_file in annotations_files:
        #         result[annotations_file] = []
        #         edf_match = pattern_edf.match(annotations_file.name)
        #         edf_number = None if not edf_match or not edf_match.group(1) else edf_match.group(1)
        #         for embedding_file in embedding_files:
        #                 embedding_match = pattern_embedding.match(embedding_file.name)
        #                 if embedding_match:

        #                         result[annotations_file].append(embedding_file)
find_matches()
        

class Embedding(Dataset):
    def __init__(
            self, path: pathlib.Path, annotations_path: pathlib.Path, annotation_schema_path: pathlib.Path
    ) -> None:
        directories = [directory for directory in path.glob("*")]
        schema = next(path.glob("*.json"))



    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
