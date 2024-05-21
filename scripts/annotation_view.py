#!/usr/bin/env python3

import mne
from pathlib import Path
import argparse


def get_unique_annotations_from_file(edf_file_path):
    """Retrieve unique annotations from a single EDF file."""
    # Read the EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

    # Extract annotations
    annotations = raw.annotations

    # Extract unique descriptions
    unique_annotations = set(annotations.description)

    return unique_annotations


def process_path(path):
    """Process a file or directory to get unique annotations."""
    unique_annotations = set()
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".edf":
        # Process a single EDF file
        unique_annotations.update(get_unique_annotations_from_file(path))
    elif path.is_dir():
        # Process all EDF files in the directory
        for edf_file in path.glob("**/*.edf"):
            unique_annotations.update(get_unique_annotations_from_file(edf_file))
    else:
        print(
            f"Path '{path}' is neither an EDF file nor a directory containing EDF files."
        )
        return

    # Print unique annotations
    for annotation in unique_annotations:
        print(annotation)


def main():
    parser = argparse.ArgumentParser(
        description="Process EDF files to retrieve unique annotations."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to an EDF file or a directory containing EDF files",
    )
    args = parser.parse_args()

    process_path(args.path)


if __name__ == "__main__":
    main()
