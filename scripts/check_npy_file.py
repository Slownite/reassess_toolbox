#!/usr/bin/env python3
import numpy as np
import argparse
import pathlib

def load_npy_file(file_path):
    try:
        data = np.load(file_path)
        print(f"Shape of the array: {data.shape}")
    except Exception as e:
        print(f"Failed to load the file. Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Load .npy files and print its shape.')
    parser.add_argument('file_path', type=pathlib.Path, help='Path to the .npy file or directory')

    args = parser.parse_args()
    if args.file_path.is_dir():
        for file in args.file_path.glob("**/*.npy"):
            load_npy_file(file)
    else:
        load_npy_file(args.file_path)

if __name__ == "__main__":
    main()
