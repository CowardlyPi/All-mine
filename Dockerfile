# Start with a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for better Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create directory for data and ensure permissions
RUN mkdir -p /mnt/data && chmod 777 /mnt/data

# Set environment variable for data directory
ENV DATA_DIR=/mnt/data

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface_hub[hf_xet]

# Add debug print statements early in the process
RUN echo "Python version:" && python --version && \
    echo "Pip version:" && pip --version

# Download transformer models at build time
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the code
COPY . .

# Print the data directory setup for debugging
RUN echo "Data directory: $DATA_DIR" && \
    ls -la $DATA_DIR

# Command to run
CMD ["python", "main.py"]
