#!/bin/bash

#=======================================================================
# PBS Pro Directives
#=======================================================================
#PBS -N MAD_exp3_strong_weak
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=96:00:00
#PBS -o job_logs/exp3.out
#PBS -e job_logs/exp3.err

# Switch to the directory where you submitted the job
cd $PBS_O_WORKDIR

echo "================================================="
echo "Job started on: $(date)"
echo "Job running on node: $(hostname)"
echo "PBS Job ID: $PBS_JOBID"
echo "Working Directory: $(pwd)"
echo "================================================="

#=======================================================================
# Environment Setup
#=======================================================================
source ~/.bashrc
conda activate agent_robustness

# AUTOMATIC PORT FINDER (Starts at 11547 to avoid common clashes)
PORT=11547
while ss -tuln | grep -q ":$PORT "; do
  PORT=$((PORT+1))
done

export MY_OLLAMA_PORT=$PORT
export OLLAMA_HOST=127.0.0.1:$MY_OLLAMA_PORT

echo "Job started on: $(date)"
echo "Using Found Port: $MY_OLLAMA_PORT"

#=======================================================================
# Ollama Server Management
#=======================================================================
echo "Starting Ollama Server..."
# Ensure the path to your ollama binary is correct
/home/vigil/24m0775/models/ollama/bin/ollama serve > ollama_exp3.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama to initialize..."
sleep 45 # Increased slightly for the 32B model to warm up drivers

# Verify connection
if ! curl -s http://localhost:$MY_OLLAMA_PORT/api/tags > /dev/null; then
    echo "CRITICAL ERROR: Ollama failed to start on port $MY_OLLAMA_PORT"
    kill $OLLAMA_PID
    exit 1
fi

#=======================================================================
# Experiment Execution
#=======================================================================
echo "Starting Asymmetric Strength Experiment (qwen3-32B-Abliterated vs llama3.1_8B)..."
echo "Starting MAD Experiment Suite..."
# --- TIMER START ---
START_TIME=$(date +%s)
echo "Starting Execution at: $(date)"

# bash scripts/run_MAD_experiments_exp3.sh
bash scripts/run_MAD_dry_run_exp3.sh

EXIT_CODE=$?

# --- TIMER END ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate hours, minutes, seconds
HOURS=$((DURATION / 3600))
MINS=$(( (DURATION % 3600) / 60 ))
SECS=$((DURATION % 60))

echo "================================================="
echo "Job finished with exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "Total Runtime: ${HOURS}h ${MINS}m ${SECS}s"
echo "================================================="

#=======================================================================
# Cleanup
#=======================================================================
echo "Cleaning up Ollama server (PID: $OLLAMA_PID)..."
kill $OLLAMA_PID
echo "Job finished with exit code $EXIT_CODE at $(date)"