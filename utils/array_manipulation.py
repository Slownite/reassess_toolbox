#!/usr/bin/env python3

import numpy as np


def pad_to_shape(array, target_shape):
    """
    Pads the input array with zeros to match the target shape.

    Parameters:
    array (np.ndarray): The input array to be padded.
    target_shape (tuple): The desired shape of the output array.

    Returns:
    np.ndarray: The padded array with the target shape.
    """
    # Ensure target shape is valid
    if any(ts < s for ts, s in zip(target_shape, array.shape)):
        raise ValueError(
            "Target shape must be greater than or equal to input shape in all dimensions."
        )

    # Create the new array with the target shape filled with zeros
    new_array = np.zeros(target_shape, dtype=array.dtype)

    # Determine the start indices for centering the original array within the new array
    start_indices = [(t - s) // 2 for t, s in zip(target_shape, array.shape)]

    # Calculate the slices for each dimension
    slices = tuple(
        slice(start, start + size) for start, size in zip(start_indices, array.shape)
    )

    # Copy the original array into the new array
    new_array[slices] = array

    return new_array
