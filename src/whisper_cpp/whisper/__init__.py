"""Python bindings for whisper.cpp using Cython."""

import importlib.metadata
import ctypes
from pathlib import Path


def get_whisper_path(): 
    return Path(__file__).parent
    

# Load libwhisper.so.1 before importing the Cython module
def load_whisper_library():
    """Load the whisper library using ctypes so it's available for the Cython module"""
    
    # Get the directory where the parent package is located
    current_dir = get_whisper_path()
    
    # Check for whisper library in current directory
    possible_paths = [
        current_dir / "libwhisper.so.1",
        current_dir / "libwhisper.so",
    ]
    
    for lib_path in possible_paths:
        if lib_path.exists():
            return ctypes.CDLL(str(lib_path))
    
    raise FileNotFoundError("Could not find libwhisper.so.1 or libwhisper.so in the package directory")

# Load the library before importing the Cython module
load_whisper_library()

__version__ = importlib.metadata.version("whisper-cpp")
__author__ = importlib.metadata.metadata("whisper-cpp")["Author"]


from .whisper_cpp import Whisper, WhisperParams, WhisperState

__all__ = ["Whisper", "WhisperParams", "WhisperState"]
