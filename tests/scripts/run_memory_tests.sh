#!/bin/bash

# Activate virtual environment
source /workspace/venv/bin/activate

# Change to the project directory
cd /workspace/Wan2.1-Enhanced

# Check if a config file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_config_file>"
  exit 1
fi

CONFIG_FILE=$1

# Run the test with the specified config file
python generate.py --config "$CONFIG_FILE" --ckpt_dir /workspace/Wan2.1-Enhanced/Wan2.1-T2V-1.3B