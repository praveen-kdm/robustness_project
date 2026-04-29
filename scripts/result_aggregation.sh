#!/bin/bash

# 1. Get the folder path passed as an argument (e.g., results/llama3.1)
TARGET_FOLDER=$1

if [ -z "$TARGET_FOLDER" ]; then
    echo "Usage: ./result_aggregation.sh <relative_path_to_folder>"
    exit 1
fi

# 2. Get the absolute path of the project root
# Since this script is in project_root/scripts, we go up one level
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. Define the path to the evaluation script and the target directory
EVAL_SCRIPT="$PROJECT_ROOT/evaluation/evaluate_result.py"
FULL_TARGET_DIR="$PROJECT_ROOT/$TARGET_FOLDER"

# Extract "llama3.1" from "results/llama3.1"
SUBFOLDER=$(basename "$TARGET_FOLDER")

# 4. Loop through all .json files in the target folder
for json_file in "$FULL_TARGET_DIR"/*.json; do
    
    # Extract just the filename (e.g., file1.json) from the full path
    filename=$(basename "$json_file")
    
    echo "Processing: $filename ..."

    # 5. Run the python command
    # We use $filename because your python script handles pathing/saving internally
    python3 "$EVAL_SCRIPT" \
        --filename "$filename" \
        --folder "$SUBFOLDER" \
        --environment multi_agent_debate
done

echo "Done! CSV files should be in $TARGET_FOLDER"
