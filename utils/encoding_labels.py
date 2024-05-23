#!/usr/bin/env python3

import numpy as np


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
