# ARGS="$@"
# python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_0 --model-client llama3.1:8b $ARGS
# python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_1 --model-client llama3.1:8b $ARGS
# python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_2 --model-client llama3.1:8b $ARGS
# python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_3 --model-client llama3.1:8b $ARGS
# python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_4 --model-client llama3.1:8b $ARGS



#!/bin/bash
# run_MAD_experiments.sh

# Get the directory where THIS script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go up one level to the project root
cd "$SCRIPT_DIR/.."

SEEDS=(42 123 999)
for SEED in "${SEEDS[@]}"
do
    for AGENT_ID in 0 1 2 3 4
    do
        # Use 'python' instead of 'python3' to ensure it uses your conda env
        python run_experiments_exp1.py \
            --environment multi_agent_debate \
            --adversarial-agent agent_$AGENT_ID \
            --model-client llama3.1:8b \
            --seed $SEED \
            --id "exp1_seed_${SEED}_agent_${AGENT_ID}"
    done
done