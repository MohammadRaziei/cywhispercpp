"""Python bindings for whisper.cpp using Cython."""

import importlib.metadata

__version__ = importlib.metadata.version("whisper-cpp")

# Import whisper submodule
from . import whisper

__all__ = ["whisper"]
