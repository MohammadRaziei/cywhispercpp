"""Basic tests for whisper-cpp bindings."""

import pytest
import whisper_cpp as wcp


class TestWhisperBasic:
    """Basic tests for Whisper class."""
    
    def test_import(self):
        """Test that we can import the module."""
        assert wcp.Whisper is not None
        assert wcp.WhisperParams is not None
        assert wcp.WhisperState is not None
    
    def test_version(self):
        """Test that version is available."""
        assert wcp.__version__ is not None
        assert isinstance(wcp.__version__, str)
        assert len(wcp.__version__) > 0
    
    def test_author(self):
        """Test that author is available."""
        assert wcp.__author__ is not None
        assert isinstance(wcp.__author__, str)
        assert len(wcp.__author__) > 0
    
    def test_whisper_class_exists(self):
        """Test that Whisper class can be instantiated (without model)."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert whisper_instance is not None
    
    def test_whisper_params_class_exists(self):
        """Test that WhisperParams class can be instantiated."""
        params = wcp.WhisperParams()
        assert params is not None
    
    def test_whisper_state_class_exists(self):
        """Test that WhisperState class can be instantiated."""
        state = wcp.WhisperState()
        assert state is not None


class TestWhisperSystemInfo:
    """Tests for system information functionality."""
    
    def test_system_info_method_exists(self):
        """Test that get_system_info method exists."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert hasattr(whisper_instance, 'get_system_info')
        assert callable(whisper_instance.get_system_info)
    
    def test_is_multilingual_method_exists(self):
        """Test that is_multilingual method exists."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert hasattr(whisper_instance, 'is_multilingual')
        assert callable(whisper_instance.is_multilingual)
    
    def test_get_vocab_size_method_exists(self):
        """Test that get_vocab_size method exists."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert hasattr(whisper_instance, 'get_vocab_size')
        assert callable(whisper_instance.get_vocab_size)
    
    def test_get_audio_ctx_size_method_exists(self):
        """Test that get_audio_ctx_size method exists."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert hasattr(whisper_instance, 'get_audio_ctx_size')
        assert callable(whisper_instance.get_audio_ctx_size)
    
    def test_get_text_ctx_size_method_exists(self):
        """Test that get_text_ctx_size method exists."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert hasattr(whisper_instance, 'get_text_ctx_size')
        assert callable(whisper_instance.get_text_ctx_size)


class TestWhisperTranscribe:
    """Tests for transcription functionality."""
    
    def test_transcribe_method_exists(self):
        """Test that transcribe method exists."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert hasattr(whisper_instance, 'transcribe')
        assert callable(whisper_instance.transcribe)
    
    def test_init_from_buffer_method_exists(self):
        """Test that init_from_buffer static method exists."""
        assert hasattr(wcp.Whisper, 'init_from_buffer')
        assert callable(wcp.Whisper.init_from_buffer)


class TestModuleStructure:
    """Tests for module structure and imports."""
    
    def test_module_has_correct_exports(self):
        """Test that module exports correct symbols."""
        expected_exports = ['Whisper', 'WhisperParams', 'WhisperState']
        for export in expected_exports:
            assert export in wcp.__all__
    
    def test_module_docstring(self):
        """Test that module has a docstring."""
        assert wcp.__doc__ is not None
        assert len(wcp.__doc__) > 0
        assert "Python bindings for whisper.cpp" in wcp.__doc__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
