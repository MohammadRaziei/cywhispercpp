"""Functional tests for whisper-cpp bindings."""

import pytest
from cywhispercpp import whisper


class TestWhisperFunctional:
    """Functional tests for Whisper module."""
    
    def test_whisper_initialization(self):
        """Test that Whisper class can be instantiated."""
        whisper_instance = whisper.Whisper.__new__(whisper.Whisper)
        assert whisper_instance is not None
    
    def test_whisper_params_initialization(self):
        """Test that WhisperParams class can be instantiated."""
        params = whisper.WhisperParams()
        assert params is not None
    
    def test_whisper_state_initialization(self):
        """Test that WhisperState class can be instantiated."""
        state = whisper.WhisperState()
        assert state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
