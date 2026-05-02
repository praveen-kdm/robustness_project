#!/bin/bash
# dry_run_exp3.sh

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# Using a single seed as requested to speed up the experiment suite
# SEED=42
SEEDS=(42)

# Define models
WEAK_MODEL="llama3.1:8b"
STRONG_MODEL="huihui_ai/qwen3-abliterated:32b"

for SEED in "${SEEDS[@]}"
do
    for AGENT_ID in 0
    do
        echo "-------------------------------------------------------"
        echo "Running Exp 3: Seed $SEED | Strong Attacker: agent_$AGENT_ID"
        echo "-------------------------------------------------------"
        
        python run_experiments_exp3.py \
            --weak-model $WEAK_MODEL \
            --strong-model $STRONG_MODEL \
            --adversarial-agent agent_$AGENT_ID \
            --seed $SEED \
            --id "strong_weak_32b_8b"
    done
done
