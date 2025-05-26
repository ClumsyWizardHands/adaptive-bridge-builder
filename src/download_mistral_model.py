from tqdm import tqdm
"""
Mistral Model Downloader

This script downloads a Mistral model in GGUF format for local use.
Models are downloaded from HuggingFace and prepared for use with the multi-model agent.
"""

import os
import sys
import argparse
import logging
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_downloader")

# URLs for Mistral models
MISTRAL_MODELS = {
    "mistral-7b-instruct-v0.2": {
        "q4": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "q5": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "q8": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf",
    },
    "mixtral-8x7b-instruct-v0.1": {
        "q4": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "q5": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
        "q8": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q8_0.gguf",
    }
}

MODEL_SIZES = {
    "mistral-7b-instruct-v0.2": {
        "q4": "4.1 GB",
        "q5": "4.8 GB",
        "q8": "7.6 GB",
    },
    "mixtral-8x7b-instruct-v0.1": {
        "q4": "26 GB",
        "q5": "31 GB", 
        "q8": "47 GB",
    }
}


def download_file(url, destination):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Destination file path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(destination):
        logger.info(f"File already exists: {destination}")
        return
    
    # Download file
    logger.info(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code != 200:
        logger.error(f"Failed to download: {response.status_code}")
        return
    
    # Get file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Create progress bar
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    logger.info(f"Download complete: {destination}")


def main():
    """Main function to download a model."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Download a Mistral model for local use")
    
    # Model selection
    parser.add_argument(
        "--model",
        choices=list(MISTRAL_MODELS.keys()),
        default="mistral-7b-instruct-v0.2",
        help="Model to download (default: mistral-7b-instruct-v0.2)"
    )
    
    # Quantization level
    parser.add_argument(
        "--quantization",
        choices=["q4", "q5", "q8"],
        default="q4",
        help="Quantization level (default: q4, smallest size)"
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory (default: models)"
    )
    
    args = parser.parse_args()
    
    # Print model information
    model_url = MISTRAL_MODELS[args.model][args.quantization]
    model_size = MODEL_SIZES[args.model][args.quantization]
    model_filename = os.path.basename(model_url)
    output_path = os.path.join(args.output_dir, model_filename)
    
    print("\n============================================")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"File size: {model_size}")
    print(f"Output path: {output_path}")
    print("============================================\n")
    
    # Confirm download
    if input("Proceed with download? (y/n): ").lower() != 'y':
        print("Download cancelled")
        return
    
    # Download the model
    download_file(model_url, output_path)
    
    # Print success message
    print("\n============================================")
    print(f"Download complete: {output_path}")
    print(f"To use this model, run: python src/run_multi_model_agent.py --model-path {output_path}")
    print("============================================\n")
    
    # Update environment variable
    if 'win' in sys.platform:
        print(f"On Windows, you can set: set LOCAL_MODEL_PATH={output_path}")
    else:
        print(f"On Linux/Mac, you can set: export LOCAL_MODEL_PATH={output_path}")


if __name__ == "__main__":
    main()