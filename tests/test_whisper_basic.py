"""Basic tests for whisper-cpp bindings."""

import pytest
import cywhisper as cw
from cywhisper import whisper


class TestWhisperBasic:
    """Basic tests for Whisper module."""
    
    def test_import(self):
        """Test that we can import the module."""
        assert whisper.Whisper is not None
        assert whisper.WhisperParams is not None
        assert whisper.WhisperState is not None
    
    def test_version(self):
        """Test that version is available."""
        assert cw.__version__ is not None
        assert isinstance(cw.__version__, str)
        assert len(cw.__version__) > 0
    
    def test_whisper_class_exists(self):
        """Test that Whisper class can be instantiated."""
        whisper_instance =  whisper.Whisper.__new__(whisper.Whisper)
        assert whisper_instance is not None
    
    def test_whisper_params_class_exists(self):
        """Test that WhisperParams class can be instantiated."""
        params = whisper.WhisperParams()
        assert params is not None
    
    def test_whisper_state_class_exists(self):
        """Test that WhisperState class can be instantiated."""
        state = whisper.WhisperState()
        assert state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
