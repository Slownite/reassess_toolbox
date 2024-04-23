import subprocess
import argparse
from pathlib import Path


def preprocessing_script(
    input_path,
    output_path,
    resolution=None,
    format=None,
    frame_rate=None,
    sampling_rate=None,
    compression=None,
):
    cmd = ["ffmpeg", "-i", input_path]

    if resolution:
        cmd += ["-vf", f"scale={resolution[0]}:{resolution[1]}"]
    if format:
        output_format = format[1]
        if output_format == ".mp4":
            cmd += [
                "-codec:v",
                "libx264",  # Spécifie le codec vidéo pour la vidéo de sortie
                "-crf",
                "23",  # Définit le facteur de qualité pour la compression (valeurs plus basses = meilleure qualité)
                "-preset",
                "fast",  # Équilibre entre vitesse de conversion et qualité
            ]
    if frame_rate:
        cmd += ["-r", str(frame_rate)]
    if sampling_rate:
        cmd += ["-ar", str(sampling_rate)]
    if compression:
        cmd += [
            "-acodec",
            compression,
        ]  # L'option spécifique dépendra du type de compression désiré

    cmd += [output_path]

    subprocess.run(cmd, check=True)


def process_directory(
    input_dir,
    output_dir,
    resolution=None,
    format=None,
    frame_rate=None,
    sampling_rate=None,
    compression=None,
):
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir_path.glob(f"**/*.{format[0]}"):
        output_file = output_dir_path / f"rgb_{input_file.name}"
        preprocessing_script(
            input_file,
            output_file,
            resolution=resolution,
            format=format,
            frame_rate=frame_rate,
            sampling_rate=sampling_rate,
            compression=compression,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for media files using ffmpeg."
    )

    parser.add_argument("input_path", type=Path, help="Input file")
    parser.add_argument("output_path", type=Path, help="Output file")
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        nargs=2,
        help="Width and height tuple[uint, uint]",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default=(".avi", ".avi"),
        nargs=2,
        help="Initial format and new format (e.g., .avi .mp4)",
    )
    parser.add_argument("-fr", "--frame_rate", type=int, help="New frame rate [uint]")
    parser.add_argument(
        "-sr", "--sampling_rate", type=int, help="New sampling rate [uint]"
    )
    parser.add_argument(
        "-c", "--compression", type=str, help="Compression type [string]"
    )
    parser.add_argument(
        "-af", "--audio_filter", type=str, help="Audio filter to apply [string]"
    )

    args = parser.parse_args()

    # Vérifier si input_path est un dossier
    if Path(args.input_path).is_dir():
        process_directory(
            args.input_path,
            args.output_path,
            resolution=args.resolution,
            format=args.format,
            frame_rate=args.frame_rate,
            sampling_rate=args.sampling_rate,
            compression=args.compression,
        )
    else:
        preprocessing_script(
            args.input_path,
            args.output_path,
            resolution=args.resolution,
            format=args.format,
            frame_rate=args.frame_rate,
            sampling_rate=args.sampling_rate,
            compression=args.compression,
        )


if __name__ == "__main__":
    main()
