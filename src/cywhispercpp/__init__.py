"""Python bindings for whisper.cpp using Cython."""

import importlib.metadata

__version__ = importlib.metadata.version("cywhispercpp")

# Import whisper submodule
from . import whisper

__all__ = ["whisper"]
