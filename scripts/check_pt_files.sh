#!/bin/bash

for file in "$@"
do
    python -c "
import torch
import sys

file_path = sys.argv[1]

try:
    torch.load(file_path)
    print(f'{file_path} is not corrupt.')
except Exception as e:
    print(f'{file_path} is corrupt. Error: {e}')
" "$file"
done

