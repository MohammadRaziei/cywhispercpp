"""Test that demonstrates C++ developers can use the pre-built whisper.cpp library."""

import os
import tempfile
import subprocess
import pytest
import sys
from pathlib import Path
from cywhispercpp.whisper import get_whisper_cpp_path


class TestCppIntegration:
    """Test C++ integration with pre-built whisper.cpp library."""
    
    def test_cpp_program_build_and_run(self):
        """Test building and running a C++ program using pre-built whisper.cpp library."""
        
        # Create a temporary directory for the test build
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"
            build_dir.mkdir()
            
            # Get the path to the pre-built whisper.cpp library using the provided function
            whisper_cpp_path = get_whisper_cpp_path()
            
            if not whisper_cpp_path.exists():
                pytest.skip("whisper.cpp path not found - run build first")
            
            # Path to test_cpp directory
            test_cpp_dir = Path(__file__).parent / "test_cpp"
            
            try:
                # Configure with CMake
                configure_result = subprocess.run(
                    [
                        "cmake", 
                        "-S", str(test_cpp_dir),
                        "-B", str(build_dir),
                        f"-DWHISPER_CPP_INSTALL_PREFIX={whisper_cpp_path}"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Check if CMake configuration was successful
                assert configure_result.returncode == 0, f"CMake configuration failed: {configure_result.stderr}"
                print("✓ CMake configuration successful")
                
                # Build the project
                build_result = subprocess.run(
                    ["cmake", "--build", str(build_dir)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                # Check if build was successful
                assert build_result.returncode == 0, f"Build failed: {build_result.stderr}"
                print("✓ Build successful")
                
                # Run the compiled program
                executable_path = build_dir / "test_whisper"
                if sys.platform == "win32":
                    executable_path = build_dir / "test_whisper.exe"
                
                run_result = subprocess.run(
                    [str(executable_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Check if program ran successfully
                assert run_result.returncode == 0, f"Program execution failed: {run_result.stderr}"
                print("✓ Program execution successful")
                
                # Analyze the output
                output = run_result.stdout
                print("Program output:")
                print(output)
                
                # Verify expected output patterns
                assert "Whisper C++ Library Test" in output
                assert "Successfully included whisper.h" in output
                assert "Whisper version:" in output
                assert "System info:" in output
                assert "Successfully created whisper parameters" in output
                assert "Successfully created whisper context parameters" in output
                assert "Maximum language ID:" in output
                assert "English language ID:" in output
                assert "Test Completed Successfully" in output
                assert "Library linking and basic functionality verified!" in output
                
                print("✓ All expected output patterns found")
                
            except subprocess.TimeoutExpired:
                pytest.fail("CMake operation timed out")
            except Exception as e:
                pytest.fail(f"Test failed with exception: {e}")
    
    def test_cmake_variables(self):
        """Test that CMake properly handles the WHISPER_CPP_INSTALL_PREFIX variable."""
        
        # Get the path to the pre-built whisper.cpp library using the provided function
        whisper_cpp_path = get_whisper_cpp_path()
        
        if not whisper_cpp_path.exists():
            pytest.skip("whisper.cpp path not found - run build first")
        
        print(f"Using whisper.cpp path: {whisper_cpp_path}")
        
        # Check that required directories exist
        assert (whisper_cpp_path / "lib").exists(), "whisper.cpp src directory not found"
        assert (whisper_cpp_path / "include").exists(), "ggml src directory not found"
        
        # Check that required libraries exist
        assert (whisper_cpp_path / "lib" / "libwhisper.so").exists(), "libwhisper.so not found"
        assert (whisper_cpp_path / "lib" / "libggml.so").exists(), "libggml.so not found"
        
        print("✓ All required libraries and directories exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
