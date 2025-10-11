"""Python bindings for whisper.cpp using Cython."""

import importlib.metadata


__version__ = importlib.metadata.version("whisper-cpp")
__author__ = importlib.metadata.metadata("whisper-cpp")["Author"]


from .whisper_c import Whisper, WhisperParams, WhisperState

__all__ = ["Whisper", "WhisperParams", "WhisperState"]
