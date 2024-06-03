import argparse
import os
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extractor script')
    parser.add_argument('-w', '--window_size', type=int, default=66, help='Window size')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('-m', '--model', default='rgb', help='Model')
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('-l', '--layer', default='Mixed_5c', help='Layer')
    parser.add_argument('--weights', help='Weights')
    parser.add_argument('source_dir', help='Source directory')
    parser.add_argument('dest_dir', help='Destination directory')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Ensure the destination directory exists
    os.makedirs(args.dest_dir, exist_ok=True)

    # Loop through all files in the source directory
    for source_file in os.listdir(args.source_dir):
        source_path = os.path.join(args.source_dir, source_file)
        dest_path = os.path.join(args.dest_dir, source_file)
        
        # Build the command to run the Python script
        cmd = [
            'python', '-m', 'scripts.I3D_extractor',
            source_path, dest_path,
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

