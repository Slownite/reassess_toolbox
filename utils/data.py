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


def downsample(dataset: Dataset, seed: int = 0, target_size: Union[int, None] = None, verbose: bool = False) -> Subset:
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
        _, label = data
        samples[label].append(i)

    # Find the size of the smallest class
    min_class = min(samples, key=lambda k: len(samples[k]))
    min_size_class = target_size or len(samples[min_class])

    # Validate target size
    if target_size and target_size > len(samples[min_class]):
        raise ValueError(
            "Target size cannot be larger than the smallest class size.")

    # Log information
    if verbose:
        logging.info(f"Smallest class: {min_class} with {
                     len(samples[min_class])} samples.")
        logging.info(f"Final subset size per class: {min_size_class}.")

    # Shuffle and sample indices
    samples = {k: random.sample(v, len(v)) for k, v in samples.items()}
    class_indices = list(chain.from_iterable(
        v[:min_size_class] for v in samples.values()))

    # Create and return the subset
    return Subset(dataset, class_indices)
