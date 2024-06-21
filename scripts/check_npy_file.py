#!/usr/bin/env python3
import numpy as np
import argparse

def load_npy_file(file_path):
    try:
        data = np.load(file_path)
        print(f"Shape of the array: {data.shape}")
    except Exception as e:
        print(f"Failed to load the file. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a .npy file and print its shape.')
    parser.add_argument('file_path', type=str, help='Path to the .npy file')

    args = parser.parse_args()
    load_npy_file(args.file_path)
