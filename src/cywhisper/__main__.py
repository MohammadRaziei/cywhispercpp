"""Main CLI entry point for whisper-cpp."""

import click
from .models.__main__ import cli as models_cli


@click.group()
@click.version_option()
def main():
    """Whisper-cpp: Python bindings for whisper.cpp with Cython."""
    pass


# Add the models subcommand
main.add_command(models_cli, name='models')


if __name__ == '__main__':
    main()
