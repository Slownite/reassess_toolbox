
import argparse
from pathlib import Path
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extractor script')
    parser.add_argument('-w', '--window_size', type=int,
                        default=66, help='Window size')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=100, help='Batch size')
    parser.add_argument('-m', '--model', default='rgb',
                        help='Model (rgb or of)')
    parser.add_argument('-nw', '--num_workers', type=int,
                        default=0, help='Number of workers')
    parser.add_argument('-l', '--layer', default='Mixed_5c', help='Layer')
    parser.add_argument('--weights', help='Weights')
    parser.add_argument('source_dir', type=Path, help='Source directory')
    parser.add_argument('dest_dir', type=Path, help='Destination directory')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Ensure the destination directory exists
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    # Set filename prefix based on model
    if args.model == 'rgb':
        file_prefix = 'rgb_'
    elif args.model == 'of':
        file_prefix = 'flow_'
    else:
        raise ValueError("Model must be either 'rgb' or 'of'")

    # Loop through all files in the source directory
    for source_file in args.source_dir.glob(f"**/{file_prefix}*.mp4"):
        dest_path = args.dest_dir / source_file.name

        # Build the command to run the Python script
        cmd = [
            'python', '-O', '-m', 'scripts.I3D_extractor',
            str(source_file), str(dest_path),
            '-w', str(args.window_size),
            '-b', str(args.batch_size),
            '-m', args.model,
            '-nw', str(args.num_workers),
            '-l', args.layer
        ]

        # Add the --weights parameter if it is specified
        if args.weights:
            cmd.extend(['--weights', args.weights])

        # Run the command
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
