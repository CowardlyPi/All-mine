import os
import sys
from pathlib import Path

print("=== DEBUG INFORMATION ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Check environment variables
print("\n=== ENVIRONMENT VARIABLES ===")
data_dir = os.getenv("DATA_DIR", "/mnt/railway/volume")
print(f"DATA_DIR: {data_dir}")
print(f"Directory exists: {Path(data_dir).exists()}")

# Check if Discord token is set (without revealing it)
discord_token = os.getenv("DISCORD_TOKEN")
print(f"DISCORD_TOKEN: {'SET' if discord_token else 'NOT SET'}")

# Check if OpenAI API key is set (without revealing it)
openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY: {'SET' if openai_api_key else 'NOT SET'}")

# Check installed packages
print("\n=== INSTALLED PACKAGES ===")
os.system("pip list")

# Check transformers
print("\n=== TRANSFORMERS CHECK ===")
try:
    from transformers import pipeline
    print("Transformers package is available")
    
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        print("Successfully loaded summarization model")
    except Exception as e:
        print(f"Error loading summarization model: {e}")
        
except ImportError as e:
    print(f"Transformers import error: {e}")

# Check volume dirs
print("\n=== VOLUME DIRECTORIES ===")
volume_path = Path(data_dir)
print(f"Volume path: {volume_path}")
print(f"Exists: {volume_path.exists()}")

if volume_path.exists():
    print("Contents:")
    try:
        for item in volume_path.iterdir():
            print(f"  {item}")
    except PermissionError:
        print("  Permission denied when trying to list directory contents")
else:
    print("Volume directory does not exist")
    try:
        print("Attempting to create it...")
        volume_path.mkdir(parents=True, exist_ok=True)
        print(f"Created {volume_path}")
    except Exception as e:
        print(f"Failed to create volume directory: {e}")

# Check for write permissions
print("\n=== WRITE PERMISSION TEST ===")
try:
    test_file = volume_path / "write_test.tmp"
    test_file.write_text("Test write access", encoding="utf-8")
    print(f"Successfully wrote to {test_file}")
    test_file.unlink()
    print("Successfully deleted test file")
except Exception as e:
    print(f"Write test failed: {e}")

print("\n=== DEBUG COMPLETE ===")
