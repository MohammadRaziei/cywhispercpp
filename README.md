# whisper-cpp

Python bindings for whisper.cpp using Cython and scikit-build.

## Features

- Python bindings for whisper.cpp speech recognition
- Built with Cython for performance
- Uses scikit-build for CMake integration
- TDD approach with pytest and pytest-xdist

## Installation

```bash
pip install .
```

For development:
```bash
pip install -e .[dev]
```

## Development

Run tests:
```bash
pytest -n auto
```

## Project Structure

```
whisper-cpp/
├── src/
│   ├── whisper_cpp/          # Python package
│   └── third_party/          # Submodules
├── tests/                    # Test files
├── CMakeLists.txt           # CMake configuration
└── pyproject.toml          # Project configuration
