#!/usr/bin/env python3
import random
import json
from more_itertools import chunked, flatten
from torch.utils.data import Dataset, Subset
def process_annotation_text_file(path: pathlib.Path, schema: dict[str, int], method_on_chunk:callable, window_size: int = 64)->list[int]:
        with open(path, 'r') as file:
            annotations = file.readlines()
        numerical_annotations = [schema.get(annotation) if schema.get(annotation) else 0 for annotation in annotations]
        return [method_on_chunk(chunk) for chunk in chunked(numerical_annotations, window_size)]

def downsample(dataset: Dataset)->Subset:
    sample = {}
    for i, (_, label) in dataset:
        if label in list(sample.key()):
            sample[label].append(i)
        else:
            sample[label] = [i]
    min_class = min(samples, key=lambda k: min(len(your_dict[k])))
    samples = {key: random.sample(v, len(v)) for k, v in samples.items()}
    class_indices = list(flatten([v[:min_class] for k, v in samples.items()]))
    subset = Subset(dataset, class_indices)
    return subset
