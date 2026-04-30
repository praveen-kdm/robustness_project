#!/bin/bash

#=======================================================================
# PBS Pro Directives
#=======================================================================
#PBS -N MAD_Replication_24m0775
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -o job_logs/
#PBS -e job_logs/

# Switch to the directory where you submitted the job
cd $PBS_O_WORKDIR

#=======================================================================
# Environment Setup
#=======================================================================
source ~/.bashrc
conda activate agent_robustness

# 1. SET THE UNIQUE PORT
# Using 11447 as planned. If you run multiple jobs, 
# you should change this for each script.
export MY_OLLAMA_PORT=11447
export OLLAMA_HOST=127.0.0.1:$MY_OLLAMA_PORT

echo "Job started on: $(date) at Port: $MY_OLLAMA_PORT"

#=======================================================================
# Ollama Server Management
#=======================================================================

# Check if port is already taken by someone else on this node
if ss -tuln | grep -q ":$MY_OLLAMA_PORT "; then
    echo "CRITICAL ERROR: Port $MY_OLLAMA_PORT is already in use. Exiting."
    exit 1
fi

echo "Starting Ollama Server..."
/home/vigil/24m0775/models/ollama/bin/ollama serve > ollama_server.log 2>&1 &

# Store the PID of Ollama to kill it later
OLLAMA_PID=$!

# Give the server time to load the GPU drivers and wake up
echo "Waiting for Ollama to initialize..."
sleep 15

# Verify connection before starting Python
if ! curl -s http://localhost:$MY_OLLAMA_PORT/api/tags > /dev/null; then
    echo "CRITICAL ERROR: Ollama failed to start on port $MY_OLLAMA_PORT"
    kill $OLLAMA_PID
    exit 1
fi

#=======================================================================
# Experiment Execution
#=======================================================================
echo "Starting Experiment..."

# Note: We pass the model client from your command line args or hardcode here
# Example: Running the 32b Qwen model
python run_experiments.py \
    --model-client qwen3:32b \
    --environment multi_agent_debate \
    --adversarial-agent agent_0 \
    --id ${PBS_JOBID%%.*}

EXIT_CODE=$?

#=======================================================================
# Cleanup
#=======================================================================
echo "Cleaning up Ollama server (PID: $OLLAMA_PID)..."
kill $OLLAMA_PID

echo "Job finished with exit code $EXIT_CODE at $(date)"