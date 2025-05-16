# Start with a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for better behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/mnt/data
ENV TRANSFORMERS_OFFLINE=0
ENV DISABLE_TRANSFORMERS=0
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create data directory with appropriate permissions
RUN mkdir -p /mnt/data && chmod 777 /mnt/data

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with fixed versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir -r requirements.txt

# Copy bot code and support files
COPY main.py bot_helper.py config.py .
COPY startup.sh .

# Make startup script executable
RUN chmod +x startup.sh

# Command to run
CMD ["./startup.sh"]
