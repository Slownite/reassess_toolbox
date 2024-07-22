#!/usr/bin/env python3
from collections import Counter
import torch
from torch.utils.data import Dataset
import pathlib
import json
from ..utils import (
    process_annotation_text_file,
)
from more_itertools import flatten
import numpy as np

def standardize(rgb, flow):
    if rgb.shape[0] > flow.shape[0]:
        new_rgb = rgb[:len(flow)]
        return new_rgb, flow
    elif rgb.shape[0] < flow.shape[0]:
        new_flow = flow[:len(rgb)]
        return rgb, new_flow
    else:
        return rgb, flow


class I3D_embeddings(Dataset):
    def __init__(
            self, path: pathlib.Path, annotation_schema_path: pathlib.Path, policy:str = "two_class_policy",
    ) -> None:
        policies = {}
        policies['two_class_policy'] = lambda x: 1 if 1 in x else 0
        with open(annotation_schema_path, 'r') as file:
            self.schema = json.load(file)

        all_files = sorted(path.glob("**/*"))
        rgb_npy_files = [f for f in all_files if f.suffix == '.npy' and f.name.startswith("rgb_")]
        flow_npy_files = [f for f in all_files if f.suffix == '.npy' and f.name.startswith("flow_")]
        txt_files = [f for f in all_files if f.suffix == '.txt']
        self.annotations = list(flatten([process_annotation_text_file(path, self.schema, policies[policy]) for path in txt_files]))
        print("len(self.annotations):", len(self.annotations))
        rgb_list = [np.load(npy_path) for npy_path in rgb_npy_files]
        flow_list = [np.load(npy_path) for npy_path in flow_npy_files]
        rgb_tensors = torch.tensor(np.concatenate(rgb_list, axis=0)).squeeze()
        flow_tensors = torch.tensor(np.concatenate(flow_list, axis=0)).squeeze()
        self.rgb_tensors, self.flow_tensors = standardize(rgb_tensors, flow_tensors)
    def __len__(self):
        return len(self.annotations)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        return (self.rgb_tensors[index], self.flow_tensors[index]), self.annotations[index]
