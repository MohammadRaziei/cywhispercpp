"""CLI interface for whisper-cpp models using Click."""

import sys
import click
from pathlib import Path
from .download_ggml_model import download_model, list_models, MODELS


@click.group()
def cli():
    """Whisper-cpp model management tools."""
    pass


@cli.command(name='download')
@click.argument('model', type=click.Choice(MODELS, case_sensitive=False))
@click.argument('models_path', required=False, type=click.Path(path_type=Path))
def download_cmd(model, models_path):
    """
    Download a Whisper model in ggml format.
    
    MODEL: Name of the model to download (e.g., tiny, base, small, medium, large-v3)
    
    MODELS_PATH: Path where to save the model (optional, defaults to current directory)
    
    Examples:
    
        whisper-cpp models download tiny
        
        whisper-cpp models download base /path/to/models
    """
    result = download_model(model, models_path, verbose=True)
    sys.exit(result)


@cli.command(name='list')
def list_cmd():
    """List all available Whisper models."""
    list_models()
    click.echo(f"\n{click.style('.en', bold=True)} = english-only  "
               f"{click.style('-q5_[01]', bold=True)} = quantized  "
               f"{click.style('-tdrz', bold=True)} = tinydiarize")


if __name__ == '__main__':
    cli()
