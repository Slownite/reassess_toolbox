#!/bin/bash

# Default values for optional parameters
WINDOW_SIZE=66
BATCH_SIZE=100
MODEL="rgb"
NUM_WORKERS=0
WEIGHTS=""
SOURCE_DIR=""
DEST_DIR=""

# List of layers
LAYERS=(
    "Conv3d_1a_7x7"
    "MaxPool3d_2a_3x3"
    "Conv3d_2b_1x1"
    "Conv3d_2c_3x3"
    "MaxPool3d_3a_3x3"
    "Mixed_3b"
    "Mixed_3c"
    "MaxPool3d_4a_3x3"
    "Mixed_4b"
    "Mixed_4c"
    "Mixed_4d"
    "Mixed_4e"
    "Mixed_4f"
    "MaxPool3d_5a_2x2"
    "Mixed_5b"
    "Mixed_5c"
)

# Function to display usage
usage() {
  echo "Usage: $0 [-w WINDOW_SIZE] [-b BATCH_SIZE] [-m MODEL] [-nw NUM_WORKERS] [--weights WEIGHTS] source_directory dest_directory"
  exit 1
}

# Parse command-line options
while getopts ":w:b:m:nw:weights:" opt; do
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

# Loop through each layer and call the process_files.sh script
for LAYER in "${LAYERS[@]}"; do
    echo "Processing layer: $LAYER"

    ./scripts/extractor.sh -w $WINDOW_SIZE -b $BATCH_SIZE -m $MODEL -nw $NUM_WORKERS -l $LAYER --weights "$WEIGHTS" "$SOURCE_DIR" "$DEST_DIR"

    if [ $? -ne 0 ]; then
        echo "Error processing layer: $LAYER" >> "${MODEL}_${LAYER}_error_log.txt"
    fi
done

echo "Processing completed. Check the *_error_log.txt files for any errors."
