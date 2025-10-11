# Whisper Model Downloader

This module is designed to download Whisper models that have been converted to GGML format.

## Usage

### Command Line Interface

After installing the package, you can use the following commands:

```bash
# List all available models
whisper-cpp models list

# Download a specific model
whisper-cpp models download tiny

# Download model to a specific path
whisper-cpp models download base /path/to/models

# Show help
whisper-cpp models --help
whisper-cpp models download --help
```

### Python API

```python
from whisper_cpp.models import download_model, list_models, MODELS

# List all models
list_models()

# Download a model
download_model("tiny")

# Download to a specific path
from pathlib import Path
download_model("base", Path("/path/to/models"))

# Access the list of available models
print(MODELS)
```

### Direct Python Module Usage

```bash
# Via models module
python -m whisper_cpp.models list
python -m whisper_cpp.models download tiny

# Or via main package
python -m whisper_cpp models list
python -m whisper_cpp models download tiny
```

## Available Models

Models are categorized as follows:

- **tiny**: Smallest model, fastest
- **base**: Base model
- **small**: Medium-sized model
- **medium**: Larger model
- **large**: Largest models (v1, v2, v3, v3-turbo)

### Model Suffixes

- `.en`: English-only models
- `-q5_0`, `-q5_1`, `-q8_0`: Quantized models (smaller but slightly less accurate)
- `-tdrz`: TinyDiarize model for speaker detection

## Features

- ✅ Download with progress indication
- ✅ Automatic detection of existing models
- ✅ Colored output support
- ✅ Error handling and cleanup of partial downloads
- ✅ Uses Python standard libraries (urllib, pathlib)
- ✅ Professional CLI interface with Click
