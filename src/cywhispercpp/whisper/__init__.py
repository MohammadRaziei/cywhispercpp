"""Python bindings for whisper.cpp using Cython."""


def get_whisper_cpp_path():
    from pathlib import Path 
    return Path(__file__).parent / "cpp" 
    

# # Load all whisper.cpp libraries before importing the Cython module
# def load_whisper_library():
#     """Load all whisper.cpp libraries using ctypes so they're available for the Cython module"""
    
#     # Get the directory where the parent package is located
#     lib_dir = get_whisper_cpp_path() / "lib"
    
#     if not lib_dir.exists():
#         logger.warning(f"Library directory does not exist: {lib_dir}")
#         return
    
#     # Load all required libraries in dependency order
#     libraries = [
#         "libggml-base.so",
#         "libggml-cpu.so",
#         "libggml.so", 
#         "libwhisper.so.1",  # Load the versioned library first
#     ]
    
#     for lib_name in libraries:
#         lib_path = lib_dir / lib_name
#         if lib_path.exists():
#             try:
#                 ctypes.CDLL(str(lib_path))
#                 logger.debug(f"Successfully loaded: {lib_name}")
#             except Exception as e:
#                 logger.warning(f"Failed to load {lib_name}: {e}")
#         else:
#             logger.warning(f"Library not found: {lib_path}")

# Load the library before importing the Cython module
# load_whisper_library()

from .cywhispercpp import Whisper, WhisperParams, WhisperState

__all__ = ["Whisper", "WhisperParams", "WhisperState"]
