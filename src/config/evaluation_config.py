"""
Evaluation Config

Specifies the hyperparameters for the evaluation process, i.e. what metrics to compute, etc.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.config._constants import MAX_SEQ_LEN


@dataclass
class PalomaEvaluationConfig:
    dataset_name: str = "pico-lm/pretokenized-paloma-tinsy"
    dataset_split: str = "val"
    max_length: int = MAX_SEQ_LEN
    batch_size: int = 16

@dataclass
class HFDatasetConfig:
    """Configuration for loading text from HuggingFace datasets."""
    name: str
    split: str = "test"
    text_field: str = "text"
    streaming: bool = True
    sample_size: Optional[int] = 1000  # Only used if streaming=True

@dataclass
class GeneralTextEvaluationConfig:
    """Configuration for evaluating perplexity on arbitrary text documents."""
    
    # Text sources (at least one must be provided)
    text_files: Optional[List[str]] = None  # Paths to text files
    text_strings: Optional[List[str]] = None  # Direct text strings
    hf_dataset: Optional[HFDatasetConfig] = None  # HuggingFace dataset config
    
    # Evaluation parameters
    max_length: int = MAX_SEQ_LEN
    batch_size: int = 16
    
    # Text processing
    chunk_size: Optional[int] = None  # Split long texts into chunks (in words)


@dataclass 
class EvaluationConfig:
    # Evaluation metrics to compute: by default, we compute the perplexity of the model on the paloma dataset
    metrics: Optional[List[str]] = field(default_factory=lambda: ["paloma"])

    # NOTE: Add other evaluation configs here
    # Each evaluation metric should have its own config
    paloma: PalomaEvaluationConfig = field(default_factory=PalomaEvaluationConfig)
    general_text: GeneralTextEvaluationConfig = field(default_factory=GeneralTextEvaluationConfig)
