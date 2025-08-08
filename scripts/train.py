#!/usr/bin/env python3
"""
A minimal script to train the Pico language model. In practice, you should just use the
`poetry run train` command to run the training pipeline. Doing so will invoke this script.
Training logic is located in `src/training/trainer.py`.
"""

from pathlib import Path
import click
import subprocess
from src.training.trainer import Trainer
from transformers import AutoTokenizer
from src.tokenizers.sentencepiece_wrapper import SentencePieceTokenizerWrapper


def register_custom_tokenizers():
    """Register custom tokenizers before any other imports."""
    try:
        
        if hasattr(AutoTokenizer, 'register'):
            AutoTokenizer.register("SentencePieceTokenizerWrapper", SentencePieceTokenizerWrapper)
            print("✅ Successfully registered SentencePieceTokenizerWrapper")
        
        # Also add to the global namespace so it can be found
        import sys
        if 'SentencePieceTokenizerWrapper' not in sys.modules:
            sys.modules['SentencePieceTokenizerWrapper'] = SentencePieceTokenizerWrapper
            
    except ImportError as e:
        print(f"⚠️  Warning: Could not register custom tokenizer: {e}")
        print("   Make sure your tokenizer file is in the Python path")

# Call this function immediately
register_custom_tokenizers()


@click.command()
@click.option(
    "--config_path",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the training configuration file",
)
def main(config_path: Path) -> None:
    """Train the Pico language model using the specified configuration."""

    print("Initializing trainer...", flush=True)
    trainer = Trainer(config_path=str(config_path))
    trainer.train()


if __name__ == "__main__":
    main()
