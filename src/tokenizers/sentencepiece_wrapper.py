"""
SentencePiece Tokenizer Wrapper for Pico Training Framework

This wrapper makes SentencePiece tokenizers compatible with the HuggingFace-style
checkpointing system by providing the necessary save_pretrained and push_to_hub methods.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
import sentencepiece as spm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer
import torch
import numpy as np

class SentencePieceTokenizerWrapper(PreTrainedTokenizer):
    """
    Wrapper for SentencePiece tokenizers to make them compatible with HuggingFace-style
    checkpointing and saving methods.
    """
    
    def __init__(
        self, 
        model_path: str, 
        vocab_size: Optional[int] = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs
    ):
        """
        Initialize the SentencePiece tokenizer wrapper.
        
        Args:
            model_path: Path to the SentencePiece model file (.model)
            vocab_size: Vocabulary size (will be inferred from model if not provided)
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
        """
        
        # Load the SentencePiece processor
        self.sp_processor = spm.SentencePieceProcessor()
        self.sp_processor.load(model_path)
        
        # Store the model path for saving later
        self.model_path = model_path
        
        # Get vocab size from the model if not provided
        if vocab_size is None:
            vocab_size = self.sp_processor.vocab_size()
        
        # Initialize the parent class with special tokens
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            vocab_size=vocab_size,
            **kwargs
        )
        
        # Set up token IDs
        self._bos_token_id = self.sp_processor.piece_to_id(bos_token)
        self._eos_token_id = self.sp_processor.piece_to_id(eos_token)
        self._unk_token_id = self.sp_processor.piece_to_id(unk_token)
        self._pad_token_id = self.sp_processor.piece_to_id(pad_token)
        
        # If pad_token doesn't exist in vocab, use unk_token
        if self._pad_token_id == self._unk_token_id:
            self._pad_token_id = self._unk_token_id
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.sp_processor.vocab_size()
    
    @property
    def bos_token_id(self) -> int:
        """Return the BOS token ID."""
        return self._bos_token_id
    
    @property 
    def eos_token_id(self) -> int:
        """Return the EOS token ID.""" 
        return self._eos_token_id
    
    @property
    def unk_token_id(self) -> int:
        """Return the UNK token ID."""
        return self._unk_token_id
    
    @property
    def pad_token_id(self) -> int:
        """Return the PAD token ID."""
        return self._pad_token_id
    
    @property
    def pad_id(self) -> int:
        """Return the PAD token ID (alternative name)."""
        return self._pad_token_id
    
    @property
    def bos_id(self) -> int:
        """Return the BOS token ID (alternative name)."""
        return self._bos_token_id
    
    @property
    def eos_id(self) -> int:
        """Return the EOS token ID (alternative name)."""
        return self._eos_token_id
    
    @property
    def unk_id(self) -> int:
        """Return the UNK token ID (alternative name)."""
        return self._unk_token_id
    
    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Tokenize text into subword pieces."""
        return self.sp_processor.encode_as_pieces(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its corresponding ID."""
        return self.sp_processor.piece_to_id(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its corresponding token."""
        return self.sp_processor.id_to_piece(index)
    
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert a list of tokens back to a string."""
        return self.sp_processor.decode_pieces(tokens)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs by adding special tokens."""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """Get special tokens mask."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, 
                token_ids_1=token_ids_1, 
                already_has_special_tokens=True
            )
        
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> list[int]:
        """Encode text to token IDs."""
        if isinstance(text, str):
            tokens = self.sp_processor.encode_as_ids(text)
        else:
            # Handle list of strings
            tokens = []
            for t in text:
                tokens.extend(self.sp_processor.encode_as_ids(t))
        
        if add_special_tokens:
            tokens = self.build_inputs_with_special_tokens(tokens)
        
        if return_tensors == "pt":
            return torch.tensor(tokens)
        elif return_tensors == "np":
            return np.array(tokens)
        
        return tokens
    
    def decode(
        self, 
        token_ids: Union[list[int], "torch.Tensor", "np.ndarray"], 
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs back to text."""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        if skip_special_tokens:
            # Remove special tokens
            special_token_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]
        
        return self.sp_processor.decode_ids(token_ids)
    
    def __call__(self, text, **kwargs):
        """Make the tokenizer callable."""
        return self.encode(text, **kwargs)
    
    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        vocab = {}
        for i in range(self.vocab_size):
            token = self.sp_processor.id_to_piece(i)
            vocab[token] = i
        return vocab
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """Save the vocabulary files."""
        # Copy the SentencePiece model file
        if filename_prefix is not None:
            model_filename = f"{filename_prefix}-sentencepiece.model"
        else:
            model_filename = "sentencepiece.model"
        
        target_model_path = os.path.join(save_directory, model_filename)
        shutil.copy2(self.model_path, target_model_path)
        
        return (target_model_path,)
    
    def save_pretrained(
        self, 
        save_directory: Union[str, os.PathLike], 
        **kwargs
    ) -> None:
        """
        Save the tokenizer to a directory.
        
        This method copies the SentencePiece model file and creates a tokenizer config
        that can be used to recreate the tokenizer.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary using parent method
        self.save_vocabulary(str(save_directory))
        
        # Create tokenizer config
        config = {
            "tokenizer_class": self.__class__.__name__,
            "auto_map": {
                "AutoTokenizer": [self.__class__.__module__, self.__class__.__name__]
            },
            "model_path": "sentencepiece.model",
            "vocab_size": self.vocab_size,
            "bos_token": str(self.bos_token),
            "eos_token": str(self.eos_token),
            "unk_token": str(self.unk_token),
            "pad_token": str(self.pad_token),
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "pad_token_id": self.pad_token_id,
        }
        
        # Save tokenizer config
        config_path = save_directory / "tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Also save a copy of this tokenizer class file for HF Hub compatibility
        tokenizer_file = save_directory / "tokenizer.py"
        import inspect
        import sys
        
        # Get the source code of this class
        try:
            source_lines = inspect.getsourcelines(self.__class__)[0]
            # Also include the imports at the top of the file
            module_source = inspect.getsource(sys.modules[self.__class__.__module__])
            
            with open(tokenizer_file, "w", encoding="utf-8") as f:
                f.write(module_source)
        except:
            # Fallback: just copy the current file
            import __main__
            if hasattr(__main__, '__file__'):
                current_file = Path(__main__.__file__).parent / "sentencepiece_tokenizer.py"
                if current_file.exists():
                    shutil.copy2(current_file, tokenizer_file)
        
        print(f"✅ Saved SentencePiece tokenizer to {save_directory}")
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tokenizer",
        revision: str = "main",
        token: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Push the tokenizer to HuggingFace Hub.
        
        Note: This creates a temporary directory, saves the tokenizer there,
        and then uploads it to the hub.
        """
        try:
            from huggingface_hub import upload_folder
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save tokenizer to temporary directory
                self.save_pretrained(temp_dir)
                
                # Upload to hub
                upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    commit_message=commit_message,
                    revision=revision,
                    token=token,
                    repo_type="model",  # Upload as part of model repo
                )
                
                print(f"✅ Pushed SentencePiece tokenizer to {repo_id}")
                
        except ImportError:
            print("⚠️  huggingface_hub not available. Cannot push to hub.")
            print("   Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"❌ Failed to push tokenizer to hub: {e}")
    
    @classmethod
    def from_pretrained(
        cls, 
        save_directory: Union[str, os.PathLike],
        **kwargs
    ) -> "SentencePieceTokenizerWrapper":
        """
        Load a tokenizer from a saved directory.
        
        Args:
            save_directory: Directory containing the saved tokenizer
            
        Returns:
            SentencePieceTokenizerWrapper instance
        """
        save_directory = Path(save_directory)
        
        # Load config
        config_path = save_directory / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"tokenizer_config.json not found in {save_directory}")
        
        # Load model
        model_path = save_directory / config["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model file not found: {model_path}")
        
        # Create tokenizer instance
        return cls(
            model_path=str(model_path),
            vocab_size=config.get("vocab_size"),
            bos_token=config.get("bos_token", "<s>"),
            eos_token=config.get("eos_token", "</s>"),
            unk_token=config.get("unk_token", "<unk>"),
            pad_token=config.get("pad_token", "<pad>"),
            **kwargs
        )


try:
    AutoTokenizer.register("SentencePieceTokenizerWrapper", SentencePieceTokenizerWrapper)
    print("✅ Registered SentencePieceTokenizerWrapper with AutoTokenizer")
except Exception as e:
    print(f"⚠️  Failed to register tokenizer: {e}")
    
# Also register for auto classes (this makes it discoverable)
SentencePieceTokenizerWrapper.register_for_auto_class()