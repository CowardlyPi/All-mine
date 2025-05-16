# Start with Python 3.10 slim image for a smaller footprint
FROM python:3.10-slim

# Set up metadata labels
LABEL maintainer="A2 Bot Developer"
LABEL description="A2 Discord Bot from NieR: Automata"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/mnt/data
ENV TRANSFORMERS_OFFLINE=0
ENV PYTHONIOENCODING=utf-8
ENV TORCH_HOME=/app/.torch

# Install system dependencies
# Note: We're adding procps to get the 'free' command for memory diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Create and set permissions for data directory
RUN mkdir -p /mnt/data && chmod 777 /mnt/data

# Copy requirements file first for better layer caching
COPY requirements.txt .

# Install Python dependencies with carefully pinned versions
# Note: We're pinning numpy to a version compatible with the transformer models
# and setting the OpenAI version to one without the 'project' parameter issue
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "openai>=1.3.5,<1.6.0" && \
    pip install --no-cache-dir "huggingface_hub[hf_xet]"

# Create helper scripts
RUN echo '#!/bin/bash\necho "Starting A2 Discord Bot"\necho "===================="\necho "Date: $(date)"\necho "Python version: $(python --version)"\necho "Available memory:"\nif command -v free &> /dev/null; then\n    free -h\nelse\n    echo "Memory stats command unavailable"\nfi\n\necho "Checking data directory: $DATA_DIR"\nif [ ! -d "$DATA_DIR" ]; then\n    echo "Creating data directory..."\n    mkdir -p "$DATA_DIR"\nfi\n\necho "Testing write permissions to data directory..."\nif touch "$DATA_DIR/write_test.tmp" && rm "$DATA_DIR/write_test.tmp"; then\n    echo "Data directory is writable."\nelse\n    echo "WARNING: Data directory is not writable. Memory features may not work."\nfi\n\necho "Starting bot..."\nexec python /app/main.py\n' > /app/startup.sh && \
    chmod +x /app/startup.sh

# Create a debug script
RUN echo '#!/usr/bin/env python\nimport os\nimport sys\nfrom pathlib import Path\nprint("=== DEBUG INFORMATION ===")\nprint(f"Python version: {sys.version}")\nprint(f"Current working directory: {os.getcwd()}")\nprint(f"Environment variables:")\nfor var in ["DISCORD_TOKEN", "OPENAI_API_KEY", "DATA_DIR"]:\n    print(f"  {var}: {'✓ SET' if os.getenv(var) else '✗ NOT SET'}")\nprint("\\nInstalled Python packages:")\nos.system("pip list")\nvolume_path = Path(os.getenv("DATA_DIR", "/mnt/data"))\nprint(f"\\nVolume path: {volume_path}")\nprint(f"Volume path exists: {volume_path.exists()}")\nif volume_path.exists():\n    try:\n        print("Contents:")\n        for item in volume_path.iterdir():\n            print(f"  {item}")\n    except Exception as e:\n        print(f"Error listing contents: {e}")\nprint("\\n=== DEBUG COMPLETE ===")\n' > /app/debug.py && \
    chmod +x /app/debug.py

# Copy main files
COPY main.py bot_helper.py config.py patch_openai.py ./

# Run the patch script to fix the OpenAI client initialization
RUN echo '#!/usr/bin/env python\nimport re\n\n# Read the main.py file\nwith open("main.py", "r") as file:\n    content = file.read()\n\n# Find and replace the OpenAI client initialization\npattern = r"self\\.openai_client = OpenAI\\(\\s*api_key=openai_api_key,\\s*organization=openai_org_id,\\s*project=openai_project_id\\s*\\)"\nreplacement = "self.openai_client = OpenAI(\\n            api_key=openai_api_key,\\n            organization=openai_org_id\\n        )"\n\n# Apply the replacement\nmodified_content = re.sub(pattern, replacement, content)\n\n# Write the modified content back to the file\nwith open("main.py", "w") as file:\n    file.write(modified_content)\n\nprint("Successfully patched main.py to fix OpenAI client initialization!")\n' > /app/patch_openai.py && \
    python /app/patch_openai.py

# Create basic healthcheck script
RUN echo '#!/usr/bin/env python\nimport sys\nimport os\n\n# Check if the Discord token is set\nif not os.getenv("DISCORD_TOKEN"):\n    print("ERROR: DISCORD_TOKEN not set")\n    sys.exit(1)\n\n# Check if the OpenAI API key is set\nif not os.getenv("OPENAI_API_KEY"):\n    print("ERROR: OPENAI_API_KEY not set")\n    sys.exit(1)\n\nprint("Basic environment checks passed")\nsys.exit(0)\n' > /app/healthcheck.py && \
    chmod +x /app/healthcheck.py

# Set up a healthcheck
HEALTHCHECK --interval=1m --timeout=30s --start-period=30s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# Command to run
CMD ["/app/startup.sh"]
