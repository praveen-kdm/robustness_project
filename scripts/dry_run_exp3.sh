#!/bin/bash
# dry_run_exp3.sh

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# Fixed variables for the dry run
SEED=42
AGENT_ID=0  # Only test position 0
WEAK_MODEL="llama3.1:8b"
STRONG_MODEL="huihui_ai/qwen3-abliterated:32b"

echo "!!! STARTING DRY RUN - SINGLE ITERATION !!!"

# We use head -n 2 (Header + 1st Row) logic inside the Python or just let 
# the loop run once by adding a sys.exit() or a flag. 
# Simplest way: Pass a specific ID flag.

python run_experiments_exp3.py \
    --weak-model $WEAK_MODEL \
    --strong-model $STRONG_MODEL \
    --adversarial-agent agent_$AGENT_ID \
    --seed $SEED \
    --id "DRY_RUN_TEST"