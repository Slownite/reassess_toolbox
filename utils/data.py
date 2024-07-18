#!/usr/bin/env python3
import random
import json
from more_itertools import chunked, flatten
import pathlib
from torch.utils.data import Dataset, Subset
def process_annotation_text_file(path: pathlib.Path, schema: dict[str, int], method_on_chunk:callable, window_size: int = 64)->list[int]:
        with open(path, 'r') as file:
            annotations = file.readlines()
        numerical_annotations = [schema.get(annotation.rstrip()) or 0 for annotation in annotations]
        reduce_annotations = [method_on_chunk(chunk) for chunk in chunked(numerical_annotations, window_size)]
        return reduce_annotations

def downsample(dataset: Dataset, seed: int = 0)->Subset:
    random.seed(seed)
    samples = {}
    for i, data in enumerate(dataset):
        _, label = data
        if label in list(samples.keys()):
            samples[label].append(i)
        else:
            samples[label] = [i]
    min_class = min(samples, key=lambda k: len(samples[k]))
    min_size_class = len(samples[min_class])
    samples = {k: random.sample(v, len(v)) for k, v in samples.items()}
    class_indices = list(flatten([v[:min_size_class] for k, v in samples.items()]))
    subset = Subset(dataset, class_indices)
    return subset
