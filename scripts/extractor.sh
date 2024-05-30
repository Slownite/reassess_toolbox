#!/bin/bash

# Default values for optional parameters
WINDOW_SIZE=66
BATCH_SIZE=256
MODEL="rgb"
NUM_WORKERS=0
LAYER="Mixed_5c"
WEIGHTS=""

# Function to display usage
usage() {
  echo "Usage: $0 [-w WINDOW_SIZE] [-b BATCH_SIZE] [-m MODEL] [-nw NUM_WORKERS] [-l LAYER] [--weights WEIGHTS] source_directory dest_directory"
  exit 1
}

# Parse command-line options
while getopts ":w:b:m:nw:l:weights:" opt; do
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
    l )
      LAYER=$OPTARG
      ;;
    weights )
      WEIGHTS=$OPTARG
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

  # Build the command to run the Python script
  CMD="python -m scripts.I3D_extractor \"$SOURCE_FILE\" \"$DEST_FILE\" -w $WINDOW_SIZE -b $BATCH_SIZE -m $MODEL -nw $NUM_WORKERS -l $LAYER"

  # Add the --weights parameter if it is specified
  if [ -n "$WEIGHTS" ]; then
    CMD+=" --weights $WEIGHTS"
  fi

  # Run the command
  eval $CMD
done
