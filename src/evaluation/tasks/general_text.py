"""
General text perplexity evaluation for any given text document.

This module provides functionality to evaluate perplexity on arbitrary text documents,
either from local files or text strings.
"""

import evaluate
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from typing import Union, List
import os

from src.config.evaluation_config import GeneralTextEvaluationConfig


def run_general_text_evaluation(
    model_path: str,
    general_text_config: GeneralTextEvaluationConfig,
) -> float:
    """Run Perplexity evaluation on arbitrary text documents.

    We use the HuggingFace evaluate library to load in and compute the perplexity metric
    on user-provided text data.

    Args:
        model_path (str): Path to the model checkpoint to be evaluated
        general_text_config (GeneralTextEvaluationConfig): Configuration for general text evaluation

    Returns:
        float: Mean perplexity score across all provided texts
    """
    
    # Validate model path exists and has required files
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Check for required files
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_path}")
    
    # Check for model weights
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
    if not has_model_file:
        raise FileNotFoundError(f"No model weights found in {model_path}. Looking for: {model_files}")
    
    disable_progress_bar()

    try:
        # Load custom evaluation space
        perplexity = evaluate.load("pico-lm/perplexity")

        # Get text data based on config
        texts = _load_texts_from_config(general_text_config)
        
        if not texts:
            raise ValueError("No text data found. Please provide either text_files, text_strings, or hf_dataset.")

        print(f"🔍 Evaluating {len(texts)} text samples with general_text metric...")

        # Compute perplexity score on the text data
        perplexity_result = perplexity.compute(
            model_id=model_path,
            predictions=texts,
            add_start_token=False,
            max_length=general_text_config.max_length,
            batch_size=general_text_config.batch_size,
            trust_remote_code=True,
        )

        mean_perplexity = perplexity_result["mean_perplexity"]

        enable_progress_bar()
        return mean_perplexity
        
    except Exception as e:
        enable_progress_bar()
        print(f"⚠ Error in general_text evaluation: {e}")
        raise


def _load_texts_from_config(config: GeneralTextEvaluationConfig) -> List[str]:
    """Load texts from various sources based on configuration.
    
    Args:
        config: Configuration specifying text sources
        
    Returns:
        List of text strings to evaluate
    """
    texts = []
    
    # Load from text files
    if config.text_files:
        for file_path in config.text_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Text file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Split large files into chunks if needed
                    if config.chunk_size and len(content.split()) > config.chunk_size:
                        chunks = _chunk_text(content, config.chunk_size)
                        texts.extend(chunks)
                    else:
                        texts.append(content)
    
    # Load from direct text strings
    if config.text_strings:
        texts.extend(config.text_strings)
    
    # Load from HuggingFace dataset
    if config.hf_dataset:
        dataset = load_dataset(
            config.hf_dataset.name, 
            split=config.hf_dataset.split,
            streaming=config.hf_dataset.streaming
        )
        
        # Extract text field
        text_field = config.hf_dataset.text_field
        if config.hf_dataset.streaming:
            # For streaming datasets, take a sample
            sample_size = config.hf_dataset.sample_size or 1000
            texts.extend([item[text_field] for _, item in zip(range(sample_size), dataset)])
        else:
            texts.extend(dataset[text_field])
    
    return texts


def _chunk_text(text: str, chunk_size: int, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of approximately chunk_size words.
    
    Args:
        text: Input text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        if end >= len(words):
            break
            
        start = end - overlap
    
    return chunks


# Alternative function for direct evaluation without config
def evaluate_text_perplexity(
    model_path: str,
    texts: Union[str, List[str]],
    max_length: int = 2048,
    batch_size: int = 16,
    chunk_size: int = None
) -> float:
    """Direct function to evaluate perplexity on text(s).
    
    Args:
        model_path: Path to model checkpoint
        texts: Single text string or list of texts
        max_length: Maximum sequence length for evaluation
        batch_size: Batch size for evaluation
        chunk_size: If provided, split texts into chunks of this size (in words)
        
    Returns:
        Mean perplexity score
    """
    # Convert single string to list
    if isinstance(texts, str):
        texts = [texts]
    
    # Chunk texts if requested
    if chunk_size:
        chunked_texts = []
        for text in texts:
            if len(text.split()) > chunk_size:
                chunked_texts.extend(_chunk_text(text, chunk_size))
            else:
                chunked_texts.append(text)
        texts = chunked_texts
    
    disable_progress_bar()
    
    # Load perplexity metric
    perplexity = evaluate.load("pico-lm/perplexity")
    
    # Compute perplexity
    result = perplexity.compute(
        model_id=model_path,
        predictions=texts,
        add_start_token=False,
        max_length=max_length,
        batch_size=batch_size,
        trust_remote_code=True,
    )
    
    enable_progress_bar()
    
    return result["mean_perplexity"]