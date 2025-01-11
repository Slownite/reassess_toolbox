
import collections
from pathlib import Path
import argparse


def calculate_overall_weights(annotation_files):
    """
    Calculate weighted coefficients for all annotations across multiple files.

    Args:
        annotation_files (list): List of paths to annotation files.

    Returns:
        dict: A dictionary with two keys ('None' and 'Other') and their corresponding weights.
    """
    total_annotations = 0
    none_count = 0
    other_count = 0

    # Process each file
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as file:
            for line in file:
                annotation = line.strip()
                total_annotations += 1
                if annotation == 'None':
                    none_count += 1
                else:
                    other_count += 1

    # Calculate weights
    weights = {
        'None': total_annotations / none_count if none_count > 0 else 0,
        'Other': total_annotations / other_count if other_count > 0 else 0
    }

    return weights


def find_annotation_files(directory):
    """
    Recursively find all .txt annotation files in a given directory.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        list: List of paths to annotation files.
    """
    directory_path = Path(directory)
    return list(directory_path.rglob("*.txt"))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate weighted coefficients for annotations in multiple files.")
    parser.add_argument("directory", type=str,
                        help="Path to the directory containing annotation files.")
    args = parser.parse_args()

    # Find all annotation files
    annotation_files = find_annotation_files(args.directory)

    if not annotation_files:
        print(f"Error: No .txt annotation files found in directory '{
              args.directory}'.")
        return

    # Calculate overall weights
    weights = calculate_overall_weights(annotation_files)

    # Print the final weights
    print("Final Calculated Weights:")
    for annotation, weight in weights.items():
        print(f"{annotation}: {weight:.4f}")


if __name__ == "__main__":
    main()
