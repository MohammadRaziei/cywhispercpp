"""Python bindings for whisper.cpp using Cython."""

import importlib.metadata
import ctypes
import os
from pathlib import Path

# Load libwhisper.so.1 before importing the Cython module
def load_whisper_library():
    """Load the whisper library using ctypes so it's available for the Cython module"""
    
    # Get the directory where this __init__.py file is located
    current_dir = Path(__file__).parent
    
    # Try different possible library locations relative to this file
    possible_paths = [
        # In the same directory as this file (where it should be installed)
        current_dir / "libwhisper.so.1",
        current_dir / "libwhisper.so",
    ]
    
    for lib_path in possible_paths:
        if lib_path.exists():
            print(f"Found whisper library at: {lib_path}")
            return ctypes.CDLL(str(lib_path))
    
    # If not found, try loading by name (might work if in LD_LIBRARY_PATH)
    try:
        return ctypes.CDLL("libwhisper.so.1")
    except OSError:
        pass
    
    try:
        return ctypes.CDLL("libwhisper.so")
    except OSError:
        pass
    
    raise FileNotFoundError("Could not find libwhisper.so.1 in any of the expected locations")

# Load the library before importing the Cython module
_ = load_whisper_library()

__version__ = importlib.metadata.version("whisper-cpp")
__author__ = importlib.metadata.metadata("whisper-cpp")["Author"]


from .whisper_cpp import Whisper, WhisperParams, WhisperState

__all__ = ["Whisper", "WhisperParams", "WhisperState"]
