
import argparse
from pathlib import Path
import mne


def extract_annotations(edf_file):
    """
    Extract annotations and their timestamps from an EDF file using MNE.

    Args:
        edf_file (str): Path to the EDF file.

    Returns:
        tuple: (duration, list of annotations with timestamps (onset, duration, annotation)).
    """
    try:
        # Load the raw EDF file with Latin-1 encoding
        raw = mne.io.read_raw_edf(
            edf_file, preload=False, verbose=False, encoding="latin1")
        annotations = raw.annotations
        # Total duration of the EDF file in seconds
        edf_duration = raw.times[-1]
    except Exception as e:
        raise ValueError(f"Failed to read EDF file: {e}")

    # Extract annotation data
    extracted_annotations = []
    for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
        extracted_annotations.append((onset, duration, description))

    return edf_duration, extracted_annotations


def display_annotations(duration, annotations):
    """
    Display the EDF duration and the extracted annotations in a readable format.

    Args:
        duration (float): Total duration of the EDF file.
        annotations (list of tuples): List of annotations with timestamps.
    """
    print(f"Total Duration of EDF File: {duration:.2f} seconds\n")
    print("Annotations extracted from the EDF file:")
    print("Onset (s)\tDuration (s)\tAnnotation")
    print("---------------------------------------")
    for onset, duration, annotation in annotations:
        print(f"{onset:.2f}\t{duration:.2f}\t{annotation}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract annotations with timestamps from an EDF file using MNE.")
    parser.add_argument("edf_file", type=str, help="Path to the EDF file.")
    args = parser.parse_args()

    edf_path = Path(args.edf_file)

    if not edf_path.is_file():
        print(
            f"Error: The file '{args.edf_file}' does not exist or is not a file.")
        return

    try:
        edf_duration, annotations = extract_annotations(args.edf_file)
        display_annotations(edf_duration, annotations)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
