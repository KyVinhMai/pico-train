"""
Pico Evaluation Package

This package implements the evaluation pipeline for the Pico language model. It provides
functionality to evaluate model performance using various metrics and handles the complete
evaluation workflow.

We recommend that each evaluation metric should have its own config, and should be
implemented as a module in the `evaluation/tasks` directory that exposes a `run_<metric_name>` function.

NOTE: Out of the box we only support Paloma, but the structure is designed to be flexible and
you are meant to add whatever metrics you want. One of the main reasons we store out
the model in the HuggingFace format is so that its easy to use third-party evaluation
libraries/frameworks.
"""

import os

import torch
from lightning.fabric import Fabric
from torch import nn

from src.config import CheckpointingConfig, EvaluationConfig

from .tasks.paloma import run_paloma_evaluation
from .tasks.general_text import run_general_text_evaluation
from src.tokenizers.sentencepiece_wrapper import SentencePieceTokenizerWrapper


def _ensure_custom_tokenizers_available():
    """Make custom tokenizers available for loading."""
    try:
        import sys
        globals()['SentencePieceTokenizerWrapper'] = SentencePieceTokenizerWrapper
        sys.modules['SentencePieceTokenizerWrapper'] = SentencePieceTokenizerWrapper
        
        print("✅ Made SentencePieceTokenizerWrapper available for evaluation")
    except ImportError:
        pass  # Custom tokenizer not available, skip registration


def run_evaluation(
    evaluation_config: EvaluationConfig,
    checkpointing_config: CheckpointingConfig,
    fabric: Fabric,
    model: nn.Module,
) -> None:
    """Run model evaluation using specified metrics in `evaluation_config`.

    This function orchestrates the complete evaluation pipeline by:
    1. Resolving the model checkpoint path (either specified or latest) to load the model from;
        during training, this is the path to the latest checkpoint in the run directory.
    2. Iterating over each evaluation metric, and running the corresponding evaluation function.
        NOTE: we suggest you follow the pattern of the evaluation functions, and implement
        your own evaluation function for each metric in the `evaluation/tasks` directory.
    3. Aggregating results across all metrics in a dictionary, and returning it.

    Args:
        evaluation_config (EvaluationConfig): Configuration object containing:
            - metrics (List[str]): Metrics to evaluate; currently supported: ["paloma", "general_text"];
            - paloma (PalomaConfig): Configuration for Paloma evaluation
            - general_text (GeneralTextEvaluationConfig): Configuration for general text evaluation
        checkpointing_config (CheckpointingConfig): Configuration object containing:
        fabric (Fabric): Lightning Fabric instance
        model (nn.Module): Original model instance

    Returns:
        Dict[str, float]: Dictionary mapping metric names to their values
            Example: {"paloma": 3.45, "general_text": 2.89}

    Raises:
        ValueError: If an unsupported evaluation metric is requested

    Example:
        results = run_evaluation(
            EvaluationConfig(
                run_name="experiment_1",
                metrics=["paloma", "general_text"],
                paloma=PalomaConfig(max_length=2048, batch_size=16),
                general_text=GeneralTextEvaluationConfig(
                    text_files=["./data/my_text.txt"],
                    batch_size=8
                )
            )
        )
    """

    _ensure_custom_tokenizers_available()

    fabric.barrier()

    model.to("cpu")  # Offloading model to CPU

    evaluation_results = {}

    # NOTE: Evaluation is only run on first processes to enable third-party evaluation libraries
    # to determine how to handle distributed evaluation.
    if fabric.global_rank == 0:
        # Construct the model path properly
        run_name = checkpointing_config.run_name
        model_path = f"{os.getcwd()}/{checkpointing_config.runs_dir}/{run_name}/{checkpointing_config.checkpoints_dir}/latest"
        os.makedirs(model_path, exist_ok=True)

        
        # Check if the model path exists and contains required files
        if not os.path.exists(model_path):
            print(f"⚠ Warning: Model path does not exist: {model_path}")
            print("Skipping evaluation - no checkpoint available yet.")
            fabric.barrier()
            model.to(fabric.device)
            return evaluation_results
        
        # Check for required HuggingFace model files
        required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"⚠ Warning: No config.json found in {model_path}")
            print("Skipping evaluation - checkpoint not ready yet.")
            fabric.barrier()
            model.to(fabric.device)
            return evaluation_results
            
        # Check if at least one model file exists
        has_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
        if not has_model_file:
            print(f"⚠ Warning: No model files found in {model_path}")
            print(f"Looking for one of: {model_files}")
            print("Skipping evaluation - model files not saved yet.")
            fabric.barrier()
            model.to(fabric.device)
            return evaluation_results

        print(f"🔍 Running evaluation with model from: {model_path}")

        for metric in evaluation_config.metrics:
            # NOTE: add your own metrics here
            if metric == "paloma":
                evaluation_result = run_paloma_evaluation(
                    model_path, evaluation_config.paloma
                )
            elif metric == "general_text":
                evaluation_result = run_general_text_evaluation(
                    model_path, evaluation_config.general_text
                )
            else:
                raise ValueError(f"Metric {metric} not supported. Currently supported: ['paloma', 'general_text']")

            evaluation_results[metric] = evaluation_result

    torch.cuda.empty_cache()

    fabric.barrier()

    model.to(fabric.device)

    return evaluation_results
