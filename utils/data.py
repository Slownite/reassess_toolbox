#!/usr/bin/env python3
import random
from more_itertools import chunked
import pathlib
from torch.utils.data import Dataset, Subset
from typing import Union
from collections import defaultdict
from itertools import chain
import logging


def process_annotation_text_file(path: pathlib.Path, schema: dict[str, int], method_on_chunk: callable, window_size: int = 64) -> list[int]:
    with open(path, 'r') as file:
        annotations = file.readlines()
    numerical_annotations = [schema.get(
        annotation.rstrip()) or 0 for annotation in annotations]
    reduce_annotations = [method_on_chunk(chunk) for chunk in chunked(
        numerical_annotations, window_size)]
    return reduce_annotations


def downsample(dataset: Dataset, seed: int = 0, target_size: int = None, verbose: bool = False) -> Subset:
    """
    Downsamples a dataset to balance classes to the size of the smallest class or a specified size.

    Args:
        dataset (Dataset): The dataset to downsample.
        seed (int): Random seed for reproducibility. Default is 0.
        target_size (int, optional): Desired number of samples per class. Defaults to the size of the smallest class.
        verbose (bool): If True, logs additional information. Default is False.

    Returns:
        Subset: A downsampled subset of the dataset with balanced classes.
    """
    random.seed(seed)
    samples = defaultdict(list)

    # Collect indices by class
    for i, data in enumerate(dataset):
        _, label = data  # Assuming dataset returns (data, label)
        samples[int(label)].append(i)  # Ensure label is an int

    # Find the size of the smallest class
    min_size_class = min(len(v) for v in samples.values())
    final_size = target_size if target_size and target_size <= min_size_class else min_size_class

    if verbose:
        for class_label, indices in samples.items():
            print(f"Class {class_label}: {len(indices)} samples")
        print(f"Downsampling to {final_size} samples per class.")

    # Shuffle and sample indices for each class
    for class_label in samples:
        random.shuffle(samples[class_label])
        samples[class_label] = samples[class_label][:final_size]

    # Flatten list of indices
    downsampled_indices = list(chain.from_iterable(samples.values()))

    return Subset(dataset, downsampled_indices)
