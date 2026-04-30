#!/bin/bash

# 1. Define your preferred port
export MY_OLLAMA_PORT=11447
export OLLAMA_HOST=127.0.0.1:$MY_OLLAMA_PORT

# 2. Check if the port is already taken by someone else
if ss -tuln | grep -q ":$MY_OLLAMA_PORT "; then
    echo "ERROR: Port $MY_OLLAMA_PORT is already in use by another user."
    echo "Please change MY_OLLAMA_PORT in this script and try again."
    exit 1
fi

echo "Starting Ollama on port $MY_OLLAMA_PORT..."

# 3. Start the Ollama server in the background
# Adjust the path below to your specific ollama binary location
/home/vigil/24m0775/models/ollama/bin/ollama serve > ollama_server.log 2>&1 &

# 4. Save the PID so we can kill it later if needed
echo $! > ollama.pid

# 5. Wait for the server to initialize (usually 5-10 seconds)
echo "Waiting for server to wake up..."
sleep 10

# 6. Verify it is running
if ss -tuln | grep -q ":$MY_OLLAMA_PORT "; then
    echo "SUCCESS: Ollama is running on localhost:$MY_OLLAMA_PORT"
    echo "You can now run: export MY_OLLAMA_PORT=$MY_OLLAMA_PORT && python run_experiments.py"
else
    echo "FAILED: Server did not start. Check ollama_server.log for details."
    exit 1
fi