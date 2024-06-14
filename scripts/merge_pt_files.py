#!/usr/bin/env python3

import argparse
import torch

def load_in_chunks(file_path, chunk_size):
    """
    Generator that yields chunks of data from a .pt file.
    """
    with open(file_path, 'rb') as f:
        while True:
            try:
                data = torch.load(f)
                yield data
            except EOFError:
                break

def save_in_chunks(data_iterator, output_path):
    """
    Save chunks of data from an iterator to a .pt file incrementally.
    """
    with open(output_path, 'wb') as f:
        for data in data_iterator:
            torch.save(data, f)

def merge_files(input_files, output_file, chunk_size):
    data_chunks = [load_in_chunks(file, chunk_size) for file in input_files]

    def merged_data():
        for chunks in zip(*data_chunks):
            merged = {}
            for key in chunks[0]:
                merged[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
            yield merged

    save_in_chunks(merged_data(), output_file)

def main():
    parser = argparse.ArgumentParser(description="Merge multiple .pt files into one.")
    parser.add_argument('input_files', nargs='+', help="List of input .pt files to merge")
    parser.add_argument('output_file', help="Output .pt file")
    parser.add_argument('--chunk_size', type=int, default=1024, help="Chunk size for processing files")

    args = parser.parse_args()

    merge_files(args.input_files, args.output_file, args.chunk_size)
    print(f"Files {args.input_files} have been merged into {args.output_file}")

if __name__ == "__main__":
    main()
