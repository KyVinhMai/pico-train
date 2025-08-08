"""
Utilities for handling custom tokenizers in the PicoLM framework.
"""

import os
from typing import Union
from pathlib import Path
from transformers import AutoTokenizer
from src.tokenizers.sentencepiece_wrapper import SentencePieceTokenizerWrapper
import sys

        
def ensure_custom_tokenizer_available():
    """
    Make sure custom tokenizer classes are available in the global namespace.
    This is a simpler approach that works with all transformers versions.
    """
    try:
        # Import your custom tokenizer class
        
        # Add to global namespace and sys.modules so it can be found by name
        globals()['SentencePieceTokenizerWrapper'] = SentencePieceTokenizerWrapper
        sys.modules['SentencePieceTokenizerWrapper'] = SentencePieceTokenizerWrapper
        
        # Also add to the current module's globals
        import __main__
        if hasattr(__main__, '__dict__'):
            __main__.__dict__['SentencePieceTokenizerWrapper'] = SentencePieceTokenizerWrapper
        
        print("✅ Made SentencePieceTokenizerWrapper globally available")
        return True
        
    except ImportError as e:
        print(f"⚠️  Could not import custom tokenizer: {e}")
        return False


def initialize_sentencepiece_tokenizer(
    model_path: str,
    vocab_size: int = None,
    **kwargs
) -> "SentencePieceTokenizerWrapper":
    """
    Initialize a SentencePiece tokenizer.
    
    Args:
        model_path: Path to the SentencePiece model file
        vocab_size: Vocabulary size (optional)
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        SentencePieceTokenizerWrapper instance
    """
    # Make sure tokenizer is available
    ensure_custom_tokenizer_available()
    
    return SentencePieceTokenizerWrapper(
        model_path=model_path,
        vocab_size=vocab_size,
        **kwargs
    )


def load_custom_tokenizer_from_checkpoint(
    checkpoint_path: Union[str, Path]
) -> Union["SentencePieceTokenizerWrapper", "PreTrainedTokenizer"]:
    """
    Load a tokenizer from a checkpoint directory.
    
    This function tries multiple approaches to load custom tokenizers.
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Check for tokenizer config
    config_path = checkpoint_path / "tokenizer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No tokenizer_config.json found in {checkpoint_path}")
    
    # Load the config to determine tokenizer class
    import json
    with open(config_path, "r") as f:
        config = json.load(f)
    
    tokenizer_class = config.get("tokenizer_class", "SentencePieceTokenizerWrapper")
    
    if tokenizer_class == "SentencePieceTokenizerWrapper":
        # Try to load our custom tokenizer
        try:
            ensure_custom_tokenizer_available()
            return SentencePieceTokenizerWrapper.from_pretrained(checkpoint_path)
        except Exception as e:
            print(f"⚠️  Failed to load custom tokenizer: {e}")
            print("   Falling back to trust_remote_code approach...")
    
    # Fallback: use AutoTokenizer with trust_remote_code
    try:
        return AutoTokenizer.from_pretrained(
            checkpoint_path, 
            trust_remote_code=True,
            use_fast=False  # Some custom tokenizers don't support fast tokenizers
        )
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        raise


def create_safe_tokenizer_config(tokenizer_instance, save_directory: Path):
    """
    Create a tokenizer config that works with both custom and standard loading.
    """
    import json
    import inspect
    import shutil
    
    # Create basic config
    config = {
        "tokenizer_class": tokenizer_instance.__class__.__name__,
        "vocab_size": tokenizer_instance.vocab_size,
        "bos_token": str(tokenizer_instance.bos_token),
        "eos_token": str(tokenizer_instance.eos_token),
        "unk_token": str(tokenizer_instance.unk_token),
        "pad_token": str(tokenizer_instance.pad_token),
        "bos_token_id": tokenizer_instance.bos_token_id,
        "eos_token_id": tokenizer_instance.eos_token_id,
        "unk_token_id": tokenizer_instance.unk_token_id,
        "pad_token_id": tokenizer_instance.pad_token_id,
    }
    
    # Add auto_map for trust_remote_code support
    if hasattr(tokenizer_instance, '__module__'):
        module_name = tokenizer_instance.__class__.__module__
        class_name = tokenizer_instance.__class__.__name__
        
        config["auto_map"] = {
            "AutoTokenizer": [f"{module_name}.{class_name}", f"{module_name}.{class_name}"]
        }
    
    # For SentencePiece tokenizer, add model path
    if hasattr(tokenizer_instance, 'model_path'):
        config["model_path"] = "sentencepiece.model"
    
    # Save config
    config_path = save_directory / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Copy tokenizer source code for trust_remote_code
    try:
        tokenizer_py = save_directory / "tokenizer.py"
        source_file = Path(inspect.getfile(tokenizer_instance.__class__))
        if source_file.exists():
            shutil.copy2(source_file, tokenizer_py)
    except Exception as e:
        print(f"⚠️  Could not copy tokenizer source: {e}")