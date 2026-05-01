#!/bin/bash

#=======================================================================
# PBS Pro Directives
#=======================================================================
#PBS -N MAD_exp2_collusion
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=96:00:00
#PBS -o job_logs/exp2.out
#PBS -e job_logs/exp2.err

# Switch to the directory where you submitted the job
cd $PBS_O_WORKDIR

echo "================================================="
echo "Experiment 2 started on: $(date)"
echo "Job running on node: $(hostname)"
echo "PBS Job ID: $PBS_JOBID"
echo "================================================="

#=======================================================================
# Environment Setup
#=======================================================================
source ~/.bashrc
conda activate agent_robustness

# 1. AUTOMATIC PORT FINDER (Offset for Exp2)
# Start at 11647 to avoid collision with Exp1 (which starts at 11547)
PORT=11647
while ss -tuln | grep -q ":$PORT "; do
  echo "Port $PORT is taken, trying $((PORT+1))..."
  PORT=$((PORT+1))
done

export MY_OLLAMA_PORT=$PORT
export OLLAMA_HOST=127.0.0.1:$MY_OLLAMA_PORT

echo "Using Unique Port for Exp2: $MY_OLLAMA_PORT"

#=======================================================================
# Ollama Server Management
#=======================================================================

echo "Starting Ollama Server for Exp2..."
# Unique log file for the Exp2 Ollama instance
/home/vigil/24m0775/models/ollama/bin/ollama serve > ollama_exp2.log 2>&1 &

OLLAMA_PID=$!

echo "Waiting for Ollama to initialize..."
sleep 15

if ! curl -s http://localhost:$MY_OLLAMA_PORT/api/tags > /dev/null; then
    echo "CRITICAL ERROR: Ollama failed to start on port $MY_OLLAMA_PORT"
    kill $OLLAMA_PID
    exit 1
fi

#=======================================================================
# Experiment Execution
#=======================================================================
echo "Starting MAD Experiment 2 (Collusion) Suite..."
START_TIME=$(date +%s)

# CALLING THE EXP2 VERSION OF THE BASH SCRIPT
bash scripts/run_MAD_experiments_exp2.sh

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(( (DURATION % 3600) / 60 ))
SECS=$((DURATION % 60))

echo "================================================="
echo "Exp2 finished with exit code $EXIT_CODE"
echo "Total Runtime: ${HOURS}h ${MINS}m ${SECS}s"
echo "================================================="

#=======================================================================
# Cleanup
#=======================================================================
echo "Cleaning up Ollama server (PID: $OLLAMA_PID)..."
kill $OLLAMA_PID