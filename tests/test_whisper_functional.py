"""Functional tests for whisper-cpp bindings."""

import pytest
import whisper_cpp as wcp


class TestWhisperFunctional:
    """Functional tests for Whisper class."""
    
    def test_whisper_initialization(self):
        """Test that Whisper class can be created."""
        # This tests that the class structure is correct
        # Note: This won't actually load a model file
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        assert whisper_instance is not None
        assert hasattr(whisper_instance, 'ctx')
    
    def test_whisper_params_initialization(self):
        """Test that WhisperParams can be created."""
        params = wcp.WhisperParams()
        assert params is not None
        assert hasattr(params, 'params')
    
    def test_whisper_state_initialization(self):
        """Test that WhisperState can be created."""
        state = wcp.WhisperState()
        assert state is not None
        assert hasattr(state, 'state')
    
    def test_whisper_method_signatures(self):
        """Test that all expected methods exist with correct signatures."""
        whisper_instance = wcp.Whisper.__new__(wcp.Whisper)
        
        # Test method existence
        methods = [
            'get_system_info',
            'is_multilingual', 
            'get_vocab_size',
            'get_audio_ctx_size',
            'get_text_ctx_size',
            'transcribe'
        ]
        
        for method in methods:
            assert hasattr(whisper_instance, method)
            assert callable(getattr(whisper_instance, method))
    
    def test_static_methods(self):
        """Test that static methods exist."""
        assert hasattr(wcp.Whisper, 'init_from_buffer')
        assert callable(wcp.Whisper.init_from_buffer)


class TestWhisperErrorHandling:
    """Tests for error handling."""
    
    def test_whisper_init_error(self):
        """Test that Whisper initialization raises error for invalid path."""
        # This should raise an error when trying to load a non-existent model
        with pytest.raises(RuntimeError):
            wcp.Whisper("/path/to/nonexistent/model.bin")
    
    def test_whisper_init_from_buffer_error(self):
        """Test that init_from_buffer raises error for invalid buffer."""
        # This should raise an error when trying to load from invalid buffer
        with pytest.raises(RuntimeError):
            wcp.Whisper.init_from_buffer(b"invalid_model_data")


class TestWhisperIntegration:
    """Integration tests for whisper-cpp."""
    
    def test_module_metadata(self):
        """Test that module metadata is accessible."""
        # Test that we can access module-level attributes
        assert hasattr(wcp, '__version__')
        assert hasattr(wcp, '__author__')
        assert hasattr(wcp, '__doc__')
        assert hasattr(wcp, '__all__')
        
        # Test that __all__ contains expected exports
        expected_exports = ['Whisper', 'WhisperParams', 'WhisperState']
        for export in expected_exports:
            assert export in wcp.__all__
    
    def test_import_alternatives(self):
        """Test different import styles."""
        # Test direct import
        from whisper_cpp import Whisper
        assert Whisper is not None
        
        # Test import with alias
        import whisper_cpp as wcp_alias
        assert wcp_alias.Whisper is not None
        
        # Test that both imports refer to the same class
        assert Whisper is wcp_alias.Whisper


@pytest.mark.skip(reason="Requires actual whisper model file")
class TestWhisperWithModel:
    """Tests that require actual whisper model files."""
    
    def test_whisper_with_valid_model(self):
        """Test whisper with a valid model file."""
        # This test is skipped by default as it requires a model file
        # model_path = "path/to/valid/model.bin"
        # whisper_instance = whisper.Whisper(model_path)
        # assert whisper_instance is not None
        pass
    
    def test_transcribe_audio(self):
        """Test audio transcription with valid model and audio."""
        # This test is skipped by default as it requires model and audio
        # whisper_instance = whisper.Whisper("path/to/model.bin")
        # audio_data = [0.0] * 16000  # 1 second of silence
        # result = whisper_instance.transcribe(audio_data)
        # assert result is not None
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
