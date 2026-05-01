#!/bin/bash
# run_MAD_experiments_exp2.sh

# Get the directory where THIS script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go up one level to the project root
cd "$SCRIPT_DIR/.."

# Since run_experiments_exp2.py now handles all 10 pairs 
# and both seeds internally, we only need to call it once.

echo "Launching Experiment 2: Multi-Agent Collusion (5-choose-2)"

python run_experiments_exp2.py \
    --environment multi_agent_debate \
    --model-client llama3.1:8b \
    --seeds 42 43 \
    --id "collusion_run"

echo "Experiment 2 Suite complete."