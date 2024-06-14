#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import pathlib
from utils import (
    one_hot_encoding,
)
import json
class Embedding(Dataset):
    def __init__(
            self, path: pathlib.Path, annotation_schema_path: pathlib.Path
    ) -> None:
        with open(path, 'r') as file:
            annotation_embedding_map = json.load(file)

        self.data_path = {k: (v[0], v[1]) for k,v in annotation_embedding_map.items()}

        with open(annotation_schema_path, 'r') as file:
            self.schema = json.load(file)

    def __len__(self):
        return self.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")

def test():
    assert extract_numbers("rgb_Beige24052003-12934503-2_1_Mixed_5c.pt") == (1,2), "the result should be (1,2)"
    assert extract_numbers("rgb_Beige24052003-12934503_1_Mixed_5c.pt") == (1, None), "the result should be (1, None)"
if __name__ == "__main__":
    test()
