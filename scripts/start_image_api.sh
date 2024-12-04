#!/bin/bash
# First time: chmod +x start_image_api.sh

# Initialize Conda (only for MacOS + Miniforge3)
# source ~/miniforge3/etc/profile.d/conda.sh

# Activate the virtual environment
# conda activate multimodal

# Export environment variables (optional if already in .env)
export API_HOST="0.0.0.0"
export API_PORT="8000"

# Start Flask API
# python -c "from src.entrypoints.api import start; start()"
uvicorn src.entrypoints.api:app --host $API_HOST --port $API_PORT --reload
