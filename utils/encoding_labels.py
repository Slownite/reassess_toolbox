#!/usr/bin/env python3

import numpy as np


def one_hot_encoding(annotations_schema: dict[str, int], key: str) -> np.ndarray:
    size = len(set(annotations_schema.values())) + 1
    one_hot_encoding = np.zeros(size)
    value = annotations_schema.get(key)
    one_hot_encoding[value] = 1
    return one_hot_encoding
