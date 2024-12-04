#!/bin/bash
# First time: chmod +x start_mlflow_tracker.sh

# Initialize Conda (only for MacOS + Miniforge3)
# Uncomment the next line if using Miniforge:
# source ~/miniforge3/etc/profile.d/conda.sh

# Activate the virtual environment
# Uncomment the next line if using Conda:
# conda activate multimodal

# Determine absolute paths
SCRIPT_DIR=$(dirname "$0")  # Current script directory
SCRIPT_DIR=$(realpath "$SCRIPT_DIR")  # Resolve to absolute path
SRC_DIR="$SCRIPT_DIR/.."  # Source directory (parent of script directory)
SRC_DIR=$(realpath "$SRC_DIR")  # Resolve to absolute path
TRACKING_DIR="$SRC_DIR/tracking"  # Tracking directory

# Export environment variables
export MLFLOW_HOST="0.0.0.0"  # Change to specific host if needed
export MLFLOW_PORT="5001"  # Port for MLflow server
export MLFLOW_BACKEND_URI="sqlite:///$TRACKING_DIR/mlflow.db"  # SQLite backend
export MLFLOW_ARTIFACT_ROOT="$TRACKING_DIR/mlruns"  # Artifact storage directory

# Set the server URI dynamically based on MLFLOW_HOST
if [ "$MLFLOW_HOST" = "0.0.0.0" ]; then
    export MLFLOW_SERVER_URI="http://localhost:${MLFLOW_PORT}"  # Localhost access
else
    export MLFLOW_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"  # Remote access
fi

# Create tracking directories if they don't exist
mkdir -p "$TRACKING_DIR"
mkdir -p "$MLFLOW_ARTIFACT_ROOT"

# Start MLflow server
echo "Starting MLflow server with the following settings:"
echo " - Backend URI: $MLFLOW_BACKEND_URI"
echo " - Artifact Root: $MLFLOW_ARTIFACT_ROOT"
echo " - Server URI: $MLFLOW_SERVER_URI"
echo " - Host: $MLFLOW_HOST"
echo " - Port: $MLFLOW_PORT"

mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT"

# Check if the server started successfully
if [ $? -ne 0 ]; then
    echo "Error: MLflow server failed to start."
    exit 1
fi

# Access the MLflow UI at the configured URI
echo "MLflow UI is accessible at: $MLFLOW_SERVER_URI"
