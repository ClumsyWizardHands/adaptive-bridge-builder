"""
Mistral Model Downloader

This script downloads a GGUF-format Mistral model for local testing.
It downloads a small quantized model to minimize download size.
"""

import os
import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file with a progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
    """
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    # Download with progress bar
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Show progress
    with open(destination, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            progress_bar.update(size)

def download_mistral_model():
    """Download a small Mistral-7B GGUF model."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Define the model to download - using a small 2-bit quantized model for testing
    model_name = "mistral-7b-instruct-v0.2.Q2_K.gguf"
    model_path = models_dir / model_name
    
    # URLs for different models - using a smaller quantized model to save space and time
    urls = {
        # Mistral 7B Instruct - smaller version, 2-bit quantized (~2.5GB)
        "mistral-7b-instruct-v0.2.Q2_K.gguf": 
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q2_K.gguf",
        
        # Alternative: Mistral 7B Instruct - 4-bit quantized (~4.5GB)
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf": 
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    }
    
    # Check if any of these models already exist
    for name in urls.keys():
        if (models_dir / name).exists():
            print(f"Found existing model: {name}")
            return models_dir / name
    
    # Download the model if none exist
    if model_name in urls:
        download_file(urls[model_name], model_path)
        print(f"Downloaded model to {model_path}")
        return model_path
    else:
        print(f"Error: Unknown model name '{model_name}'")
        return None

def download_dummy_model():
    """Create a dummy model file for testing without actually downloading a large model."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    dummy_file = models_dir / "dummy-mistral-model.gguf"
    
    if not dummy_file.exists():
        print(f"Creating dummy model file for testing: {dummy_file}")
        # Create a small binary file with a GGUF-like header
        with open(dummy_file, 'wb') as f:
            # GGUF magic bytes and a small amount of data
            f.write(b'GGUF\x00\x00\x00\x00\x01\x00\x00\x00This is a dummy model file for testing.')
    else:
        print(f"Dummy model file already exists: {dummy_file}")
    
    return dummy_file

if __name__ == "__main__":
    print("Mistral Model Downloader")
    print("------------------------")
    
    choice = input("Do you want to:\n1. Download a real model (~2.5GB)\n2. Create a dummy model file\nChoice [2]: ")
    if choice == "1":
        model_path = download_mistral_model()
    else:
        model_path = download_dummy_model()
    
    if model_path:
        print("\nDone! You can now use this model with the LLM integration test:")
        print(f"Set LOCAL_MODEL_PATH={model_path}")
