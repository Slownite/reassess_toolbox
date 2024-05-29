#!/usr/bin/env sh

#!/bin/bash

# Default values for optional parameters
WINDOW_SIZE=66
BATCH_SIZE=256
MODEL="rgb"
NUM_WORKERS=0

# Function to display usage
usage() {
  echo "Usage: $0 [-w WINDOW_SIZE] [-b BATCH_SIZE] [-m MODEL] [-nw NUM_WORKERS] source_directory dest_directory"
  exit 1
}

# Parse command-line options
while getopts ":w:b:m:nw:" opt; do
  case ${opt} in
    w )
      WINDOW_SIZE=$OPTARG
      ;;
    b )
      BATCH_SIZE=$OPTARG
      ;;
    m )
      MODEL=$OPTARG
      ;;
    nw )
      NUM_WORKERS=$OPTARG
      ;;
    \? )
      usage
      ;;
  esac
done
shift $((OPTIND -1))

# Check for the required positional arguments
if [ "$#" -ne 2 ]; then
    usage
fi

# Assign positional arguments to variables
SOURCE_DIR=$1
DEST_DIR=$2

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Loop through all files in the source directory
for SOURCE_FILE in "$SOURCE_DIR"/*; do
  # Extract the filename without the path
  FILE_NAME=$(basename "$SOURCE_FILE")
  # Define the destination file path
  DEST_FILE="$DEST_DIR/$FILE_NAME"

  # Apply the I3D_extractor.py script to the source file and save to the destination file
  python -m scripts.I3D_extractor -w $WINDOW_SIZE -b $BATCH_SIZE -m $MODEL -nw $NUM_WORKERS "$SOURCE_FILE" "$DEST_FILE"
done
