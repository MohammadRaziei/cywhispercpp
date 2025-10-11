# whisper-cpp

Python bindings for whisper.cpp using Cython and scikit-build.

## Features

- Python bindings for whisper.cpp speech recognition
- Built with Cython for performance
- Uses scikit-build for CMake integration
- TDD approach with pytest and pytest-xdist
- Built-in model downloader with CLI

## Installation

```bash
pip install .
```

For development:
```bash
pip install -e .[dev]
```

## Quick Start

### Download Models

```bash
# List available models
whisper-cpp models list

# Download a model
whisper-cpp models download tiny

# Download to specific path
whisper-cpp models download base /path/to/models
```

### Use in Python

```python
from whisper_cpp.models import download_model

# Download a model
download_model("tiny")
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
│   │   ├── models/           # Model downloader module
│   │   └── __main__.py       # CLI entry point
│   └── third_party/          # Submodules
├── tests/                    # Test files
├── CMakeLists.txt           # CMake configuration
└── pyproject.toml          # Project configuration
```
