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

echo "All memory tests finished. Generating analysis report..."

# Generate the final report
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="enhanced_memory_analysis_report_${TIMESTAMP}.md"
python3 analyzer.py --log_file profiler_logs/baseline/memory_usage.log --output_file "$REPORT_FILE"

echo "Memory analysis report generated at $REPORT_FILE"