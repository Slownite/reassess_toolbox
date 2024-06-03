import argparse
import os
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract all layers script')
    parser.add_argument('-w', '--window_size', type=int, default=66, help='Window size')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('-m', '--model', default='rgb', help='Model')
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--weights', help='Weights')
    parser.add_argument('source_dir', help='Source directory')
    parser.add_argument('dest_dir', help='Destination directory')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    layers = [
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c"
    ]

    for layer in layers[::-1]:
        print(f"Processing layer: {layer}")
        
        cmd = [
            './extractor.py',  # Adjust this to the correct path if needed
            '-w', str(args.window_size),
            '-b', str(args.batch_size),
            '-m', args.model,
            '-nw', str(args.num_workers),
            '-l', layer,
            args.source_dir,
            args.dest_dir
        ]
        
        if args.weights:
            cmd.extend(['--weights', args.weights])
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            with open(f"{args.model}_{layer}_error_log.txt", 'a') as log_file:
                log_file.write(f"Error processing layer: {layer}\n")
    
    print("Processing completed. Check the *_error_log.txt files for any errors.")

if __name__ == '__main__':
    main()

