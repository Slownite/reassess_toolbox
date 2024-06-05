#!/usr/bin/env python3
from argparse import ArgumentParser
import pathlib
import mne

def extract_edf_annotations(path: pathlib.Path) -> list[str]:
    # Read the EDF file
    raw = mne.io.read_raw_edf(str(path), encoding="latin1")

    # Get annotations
    raw_annotations = raw.annotations
    duration_seconds = raw.times[-1]
    duration_frames = int(duration_seconds * 25)
    annotations = [None for i in range(duration_frames)]
    # Extract information from each annotation
    for annot in raw_annotations:
        onset = int(annot["onset"] * 25)
        description = annot["description"]
        annotations[onset] = description
    return annotations


def annotations_to_text(annotations: list[None | str], path: pathlib.Path) -> None:
    data = "\n".join([str(a) for a in annotations])
    with path.open(mode="w") as file:
        file.write(data)
    return


def multiple_annotations_file_to_text(
    src_path: pathlib.Path, dest_path: pathlib.Path, output: str
):
    for path in src_path.glob("**/*.edf"):
        annotations = extract_edf_annotations(path)
        annotations_to_text(annotations, dest_path / f"{output}_{path.stem}_{dest_path.stem}.txt")


def generate_annotations(
    src_path: pathlib.Path, dest_path: pathlib.Path, output: str
) -> None:
    if src_path.is_file() and dest_path.is_file():
        annotations = extract_edf_annotations(src_path)
        annotations_to_text(
            annotations, dest_path.parent / f"{output}_{src_path}_{dest_path.stem}"
        )

    elif src_path.is_dir() and dest_path.is_dir():
        multiple_annotations_file_to_text(src_path, dest_path, output)
    else:
        raise ValueError("make sure both path are directories or both files")


def main() -> None:
    parser = ArgumentParser(
        prog="annotations extractor",
        description="programme d'extraction d'annotations depuis un fichier EDF",
    )
    parser.add_argument("src_path", type=pathlib.Path)
    parser.add_argument("dest_path", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=str, default="annotations")
    args = parser.parse_args()
    generate_annotations(args.src_path, args.dest_path, args.output)

if __name__ == "__main__":
    main()
