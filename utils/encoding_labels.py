import numpy as np
import torch
import pathlib
def read_annotations(file_path: pathlib.Path, window_size: int)-> list:
    block = []
    for _ in range(window_size):
        with open(file_path, "r") as file:
            block.append(file.readline())
        yield block

def read_embeddings(file_path: pathlib.Path)-> torch.Tensor:
    with open(file_path, "r") as file:


def one_hot_encoding(annotations_schema: dict[str, int], keys: list[str]) -> np.ndarray:
    size = len(set(annotations_schema.values())) + 1
    encoding = np.zeros(size)
    for key in keys:
        value = annotations_schema.get(key)
        encoding[value] = 1
    if np.all(one_hot_encoding[1:] == 0):
        encoding[0] = 1
    else:
        encoding[0] = 0
    return one_hot_encoding
