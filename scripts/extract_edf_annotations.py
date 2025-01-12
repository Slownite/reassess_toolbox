
import argparse
from pathlib import Path
import pyedflib


def extract_annotations(edf_file):
    """
    Extract annotations and their timestamps from an EDF file.

    Args:
        edf_file (str): Path to the EDF file.

    Returns:
        list of tuples: A list of annotations with timestamps (onset, duration, annotation).
    """
    # Open the EDF file
    try:
        edf = pyedflib.EdfReader(edf_file)
        annotations = edf.readAnnotations()
        edf.close()
    except Exception as e:
        raise ValueError(f"Failed to read EDF file: {e}")

    # Extract annotation data
    extracted_annotations = []
    for onset, duration, annotation in zip(annotations[0], annotations[1], annotations[2]):
        extracted_annotations.append((onset, duration, annotation))

    return extracted_annotations


def display_annotations(annotations):
    """
    Display the extracted annotations in a readable format.

    Args:
        annotations (list of tuples): List of annotations with timestamps.
    """
    print("Annotations extracted from the EDF file:")
    print("Onset (s)\tDuration (s)\tAnnotation")
    print("---------------------------------------")
    for onset, duration, annotation in annotations:
        print(f"{onset:.2f}\t{duration:.2f}\t{annotation}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract annotations with timestamps from an EDF file.")
    parser.add_argument("edf_file", type=str, help="Path to the EDF file.")
    args = parser.parse_args()

    edf_path = Path(args.edf_file)

    if not edf_path.is_file():
        print(f"Error: The file '{
              args.edf_file}' does not exist or is not a file.")
        return

    try:
        annotations = extract_annotations(args.edf_file)
        if annotations:
            display_annotations(annotations)
        else:
            print("No annotations found in the EDF file.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
