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
python generate.py --config "$CONFIG_FILE"

echo "All memory tests finished. Generating analysis report..."

# Extract the test name from the config file
TEST_NAME=$(grep 'name:' "$CONFIG_FILE" | awk -F '"' '{print $2}')

# Generate the final report
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="enhanced_memory_analysis_report_${TEST_NAME}_${TIMESTAMP}.md"

# Find the latest memory events file for the current test
LATEST_EVENTS_FILE=$(find profiler_logs -name "memory_events_${TEST_NAME}_*.json" -print0 | xargs -0 ls -t | head -n 1)

if [ -z "$LATEST_EVENTS_FILE" ]; then
    echo "Error: No memory events file found for test '$TEST_NAME'."
    exit 1
fi

python3 tests/scripts/analyzer.py --input_file "$LATEST_EVENTS_FILE" --output_file "$REPORT_FILE"


echo "Memory analysis report generated at $REPORT_FILE"