from pathlib import Path
import numpy as np
import argparse


def concatenate_npy(output_file: str, input_files: list):
    """
    Concatenate multiple .npy files into a single .npy file.

    Parameters:
    - output_file (str): Path to the output .npy file.
    - input_files (list): List of paths to input .npy files.
    """
    # Convert input files to Path objects
    input_paths = [Path(file) for file in input_files]

    # Check if all input files exist
    for file in input_paths:
        if not file.exists():
            raise FileNotFoundError(f"Input file not found: {file}")

    # Load and concatenate .npy files
    try:
        arrays = [np.load(file) for file in input_paths]
        concatenated_array = np.concatenate(arrays, axis=0)
        np.save(output_file, concatenated_array)
        print(f"Concatenation complete. Output file: {output_file}")
    except Exception as e:
        print(f"Error during concatenation: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple .npy files into one.")
    parser.add_argument("output_file", type=str,
                        help="Path to the output .npy file.")
    parser.add_argument("input_files", type=str, nargs='+',
                        help="Paths to input .npy files.")

    args = parser.parse_args()

    try:
        concatenate_npy(args.output_file, args.input_files)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
