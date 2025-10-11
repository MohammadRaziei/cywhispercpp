#!/usr/bin/env python3
"""
This script downloads Whisper model files that have already been converted to ggml format.
This way you don't have to convert them yourself.
"""

import argparse
import sys
import urllib.request
import shutil
from pathlib import Path
from typing import List


# ANSI color codes
BOLD = "\033[1m"
RESET = "\033[0m"

# Source URLs
DEFAULT_SRC = "https://huggingface.co/ggerganov/whisper.cpp"
DEFAULT_PFX = "resolve/main/ggml"

# Available Whisper models
MODELS = [
    "tiny",
    "tiny.en",
    "tiny-q5_1",
    "tiny.en-q5_1",
    "tiny-q8_0",
    "base",
    "base.en",
    "base-q5_1",
    "base.en-q5_1",
    "base-q8_0",
    "small",
    "small.en",
    "small.en-tdrz",
    "small-q5_1",
    "small.en-q5_1",
    "small-q8_0",
    "medium",
    "medium.en",
    "medium-q5_0",
    "medium.en-q5_0",
    "medium-q8_0",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v2-q8_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
]


def list_models() -> None:
    """List all available models grouped by class."""
    print("\nAvailable models:")
    model_class = ""
    for model in MODELS:
        # Get the model class (first part before . or -)
        this_model_class = model.split('.')[0].split('-')[0]
        if this_model_class != model_class:
            print("\n ", end="")
            model_class = this_model_class
        print(f" {model}", end="")
    print("\n")


def download_with_progress(url: str, output_path: Path) -> bool:
    """
    Download a file from URL to output_path with progress indication.
    
    Args:
        url: URL to download from
        output_path: Path where to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading from: {url}")
        
        # Open the URL
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            with open(output_path, 'wb') as out_file:
                downloaded = 0
                block_size = 8192
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                    else:
                        print(f"\rDownloaded: {downloaded} bytes", end="")
                
                print()  # New line after progress
        
        return True
        
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False


def get_default_download_path() -> Path:
    """Get the default download path based on script location."""
    script_path = Path(__file__).resolve().parent
    
    # Check if the script is inside a /bin/ directory
    if script_path.name == "bin":
        return Path.cwd()
    else:
        return script_path


def download_model(model: str, models_path: Path = None) -> int:
    """
    Download a specific Whisper model.
    
    Args:
        model: Model name to download
        models_path: Path where to save the model (optional)
        
    Returns:
        0 if successful, 1 otherwise
    """
    # Validate model
    if model not in MODELS:
        print(f"Invalid model: {model}")
        list_models()
        return 1
    
    # Set default path if not provided
    if models_path is None:
        models_path = get_default_download_path()
    
    # Ensure models_path is a Path object
    models_path = Path(models_path)
    
    # Create directory if it doesn't exist
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Check if model contains 'tdrz' and update the src and pfx accordingly
    src = DEFAULT_SRC
    pfx = DEFAULT_PFX
    
    if "tdrz" in model:
        src = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
        pfx = "resolve/main/ggml"
    
    # Construct output filename
    output_file = models_path / f"ggml-{model}.bin"
    
    # Check if model already exists
    if output_file.exists():
        print(f"Model {model} already exists. Skipping download.")
        return 0
    
    # Download the model
    print(f"Downloading ggml model {model} from '{src}' ...")
    
    url = f"{src}/{pfx}-{model}.bin"
    
    if not download_with_progress(url, output_file):
        print(f"Failed to download ggml model {model}")
        print("Please try again later or download the original Whisper model files and convert them yourself.")
        # Clean up partial download
        if output_file.exists():
            output_file.unlink()
        return 1
    
    # Success message
    print(f"\nDone! Model '{model}' saved in '{output_file}'")
    print("You can now use it like this:\n")
    
    # Check for whisper-cli command
    whisper_cmd = shutil.which("whisper-cli")
    if whisper_cmd is None:
        whisper_cmd = "./build/bin/whisper-cli"
    
    print(f"  $ {whisper_cmd} -m {output_file} -f samples/jfk.wav\n")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Whisper model files in ggml format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"{BOLD}.en{RESET} = english-only  {BOLD}-q5_[01]{RESET} = quantized  {BOLD}-tdrz{RESET} = tinydiarize"
    )
    
    parser.add_argument(
        "model",
        help="Model name to download"
    )
    
    parser.add_argument(
        "models_path",
        nargs="?",
        default=None,
        help="Path where to save the model (default: script directory or current directory if in /bin/)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        list_models()
        return 0
    
    # Download the model
    return download_model(args.model, args.models_path)


if __name__ == "__main__":
    sys.exit(main())
