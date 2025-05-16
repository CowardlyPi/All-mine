#!/bin/bash

# Print environment information
echo "Starting A2 Discord Bot"
echo "======================="
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "DATA_DIR: $DATA_DIR"

# Run debug script if needed
if [ "$DEBUG" = "true" ]; then
  echo "Running debug script..."
  python DEBUG.py
fi

# Start the bot
echo "Starting bot..."
python main.py
