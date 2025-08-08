"""
Utilities for initializing components of the training process.

Here, we initialize all of the components that are part of the learning process. From logging,
and checkpointing to the optimizer to the dataset and the dataloader, this file contains the
logic for setting up the classes and functions that are used in the training loop.

As always, this code is meant to be basic. We hard-code the obvious defaults, and leave the
more experimental stuff to you.
"""

import logging
import os
import warnings
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Dict, Optional, Union


import lightning as L
import torch
import wandb
import yaml
from datasets import Dataset, DownloadConfig, load_dataset
from datasets import config as datasets_config
from huggingface_hub import add_collection_item, create_branch, create_repo
from lightning.fabric.loggers import Logger as FabricLogger
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from wandb.integration.lightning.fabric import WandbLogger
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence


from src.config import (
    CheckpointingConfig,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    MonitoringConfig,
    TrainingConfig,
)
from src.model import PicoDecoder, initialize_gpt2_model
from src.training.utils.io import use_backoff
from src.training.utils.tokenizer_utils import initialize_sentencepiece_tokenizer

warnings.filterwarnings(
    "ignore",
    message=".*This integration is tested and supported for lightning Fabric.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Please report any issues to.*",
)

########################################################
#
# Basic Initialization
#
########################################################


def _apply_config_overrides(config, overrides: dict):
    """Recursively apply configuration overrides to a dataclass config object.

    Args:
        config: Base configuration object (must be a dataclass)
        overrides: Dictionary of override values matching config structure

    Returns:
        Modified config object with overrides to the config.
    """
    for field in fields(config):
        field_value = getattr(config, field.name)
        if is_dataclass(field_value):
            _apply_config_overrides(field_value, overrides.get(field.name, {}))
        else:
            if field.name in overrides:
                setattr(config, field.name, overrides[field.name])
    return config


def initialize_configuration(
    config_path: Optional[str] = None,
) -> Dict[
    str,
    Union[
        DataConfig,
        ModelConfig,
        TrainingConfig,
        EvaluationConfig,
        MonitoringConfig,
        CheckpointingConfig,
    ],
]:
    """Initialize configuration objects with optional overrides from a YAML file.

    This function initializes all of the configuration objects, and then applies
    any overrides from the config_path file. If no config_path is provided,
    the function will use the default configuration objects.

    Args:
        config_path: Path to a YAML file containing configuration overrides.

    Returns:
        A dictionary containing the initialized configuration objects.
    """
    data_config = DataConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    evaluation_config = EvaluationConfig()
    monitoring_config = MonitoringConfig()
    checkpointing_config = CheckpointingConfig()

    if config_path:
        overrides = yaml.safe_load(open(config_path, "r"))
        data_config = _apply_config_overrides(data_config, overrides.get("data", {}))
        model_config = _apply_config_overrides(model_config, overrides.get("model", {}))
        training_config = _apply_config_overrides(
            training_config, overrides.get("training", {})
        )
        evaluation_config = _apply_config_overrides(
            evaluation_config, overrides.get("evaluation", {})
        )
        monitoring_config = _apply_config_overrides(
            monitoring_config, overrides.get("monitoring", {})
        )
        checkpointing_config = _apply_config_overrides(
            checkpointing_config, overrides.get("checkpointing", {})
        )

    configs = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "evaluation": evaluation_config,
        "monitoring": monitoring_config,
        "checkpointing": checkpointing_config,
    }

    return configs


def initialize_run_dir(checkpointing_config: CheckpointingConfig) -> str:
    """Initialize a directory for the current training run.

    Creates a unique directory for storing training, evaluation, and logging artifacts.
    If no run name is specified in the config, generates a timestamp-based name.

    Args:
        checkpointing_config: Configuration object containing run settings.
            NOTE: Must have a 'run_name' attribute that can be None, in which case
            a timestamp-based name will be generated.

    Returns:
        str: The path to the run directory.
    """
    run_name = checkpointing_config.run_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpointing_config.run_name = run_name

    run_dir = os.path.join(checkpointing_config.runs_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def initialize_fabric(
    training_config: TrainingConfig, wandb_logger: Optional[FabricLogger] = None
):
    """Initialize Lightning Fabric for distributed training.

    Sets up a Lightning Fabric instance with the specified configuration for
    handling distributed training, mixed precision, and logging.

    Args:
        training_config: Configuration object containing fabric settings
            (accelerator, precision, devices, etc.).
        wandb_logger: Optional weights and biases logger instance for experiment tracking

    Returns:
        L.Fabric: Initialized Lightning Fabric instance.

    Example:
        >>> fabric = initialize_fabric(training_config, wandb_logger)
    """

    total_devices = (
        training_config.fabric.num_devices * training_config.fabric.num_nodes
    )

    if total_devices > 1:
        strategy = "deepspeed_stage_2"
    else:
        strategy = "auto"  # Sets up SingleDevice Strategy by default

    # NOTE: The strategy is set to use either DeepSpeed (Zero Stage 2) on multi-GPU,
    # or SingleDevice Strategy on single-GPU set ups. If you'd like to use a different strategy,
    # you can change the strategy flag in the fabric initialization, but be aware that this might
    # cause issues with checkpointing, evaluation, etc.

    fabric = L.Fabric(
        accelerator=training_config.fabric.accelerator,
        precision=training_config.fabric.precision,
        devices=training_config.fabric.num_devices,
        num_nodes=training_config.fabric.num_nodes,
        loggers=[wandb_logger] if wandb_logger is not None else None,
        strategy=strategy,
    )

    fabric.launch()

    return fabric


########################################################
#
# Dataset and Tokenization Initialization
#
########################################################


# @use_backoff(max_retries=20)
def initialize_dataset(
    data_config: DataConfig,
    fabric: L.Fabric,
    initial_batch_step: Optional[int] = 0,
    return_fast_forward_steps: bool = False,
):
    """Initialize dataset based on the given config."""
    from .data import ShardedIterableDataset

    datasets_config.STREAMING_READ_MAX_RETRIES = 40
    datasets_config.STREAMING_READ_RETRY_INTERVAL = 10
    download_config = DownloadConfig(max_retries=20)
    
    fast_forward_steps = 0

    print("Attempting to load dataset!...", flush=True)

    if data_config.type == "huggingface":
        print("Loading HuggingFace dataset...", flush=True)

        if data_config.dataset.name == "pico-lm/pretokenized-dolma":
            # Handle Dolma dataset with sharding logic
            if initial_batch_step is not None:
                examples_per_shard = 20_480
                total_shards = 10_000
                batches_per_shard = examples_per_shard // data_config.dataloader.batch_size
                shard_idx = initial_batch_step // batches_per_shard

                data_files = [
                    f"data/train-{str(_shard_idx).zfill(5)}-of-{total_shards}.parquet"
                    for _shard_idx in range(shard_idx, total_shards)
                ]
                fast_forward_steps = initial_batch_step % batches_per_shard
            else:
                data_files = None

            base_dataset = load_dataset(
                data_config.dataset.name,
                split="train",
                streaming=True,
                data_files=data_files,
                download_config=download_config,
            )
            
            train_dataset = ShardedIterableDataset(
                base_dataset, fabric.global_rank, fabric.world_size
            )
            val_dataset = None  # Dolma doesn't have a validation split in this setup
        else:
            # Other HuggingFace datasets
            base_dataset = load_dataset(
                data_config.dataset.name,
                split="train",
                streaming=True,
                download_config=download_config,
            )
            train_dataset = base_dataset
            val_dataset = None

    else:  # Local dataset
        print("Loading local dataset...", flush=True)
        
        # Load local text files
        try:
            # Use the 'text' loader for plain text files
            dataset_dict = load_dataset(
                'text', 
                data_files={
                    'train': data_config.dataset.train_dataset.path_id, 
                    'test': data_config.dataset.val_dataset.path_id
                },
                streaming=False,  # Load into memory for local files
                download_config=download_config,
            )
            
            train_ds = dataset_dict['train']
            test_ds = dataset_dict['test'] if 'test' in dataset_dict else None
            
            print(f"✅ Loaded train dataset with {len(train_ds)} examples", flush=True)
            if test_ds:
                print(f"✅ Loaded test dataset with {len(test_ds)} examples", flush=True)
            
            # Debug: Print a few samples to check format
            print("📋 Sample data format:", flush=True)
            for i in range(min(3, len(train_ds))):
                sample = train_ds[i]
                print(f"  Sample {i}: {type(sample)} - Keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
                if isinstance(sample, dict) and 'text' in sample:
                    text_preview = sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text']
                    print(f"    Text preview: {repr(text_preview)}")
                else:
                    print(f"    Raw sample: {sample}")
            
            # # Convert to iterable datasets for distributed training
            # if data_config.streaming or len(train_ds) > 100000:  # Use streaming for large datasets
            #     print("🔄 Converting to streaming datasets for distributed training...", flush=True)
            #     train_dataset = ShardedIterableDataset(
            #         train_ds.to_iterable_dataset(), fabric.global_rank, fabric.world_size
            #     )
            #     val_dataset = ShardedIterableDataset(
            #         test_ds.to_iterable_dataset(), fabric.global_rank, fabric.world_size
            #     ) if test_ds else None
            # else:
            #     # For smaller datasets, we can keep them as regular datasets
            #     train_dataset = train_ds
            #     val_dataset = test_ds

            print("🔄 Converting to streaming datasets for distributed training...", flush=True)
            train_dataset = ShardedIterableDataset(
                train_ds.to_iterable_dataset(), fabric.global_rank, fabric.world_size
            )
            val_dataset = ShardedIterableDataset(
                test_ds.to_iterable_dataset(), fabric.global_rank, fabric.world_size
            ) if test_ds else None
        
            
        except Exception as e:
            print(f"❌ Error loading local dataset: {e}", flush=True)
            print(f"   Train path: {data_config.dataset.train_dataset.path_id}")
            print(f"   Val path: {data_config.dataset.val_dataset.path_id}")
            raise
    
    print("✅ Dataset loading complete!", flush=True)
    
    if return_fast_forward_steps:
        return train_dataset, val_dataset, fast_forward_steps
    else:
        return train_dataset, val_dataset


# def initialize_tokenizer(data_config: DataConfig):
#     """Initialize the tokenizer for text processing.

#     This function can be extended to include custom tokenization logic.

#     Args:
#         data_config: Configuration object containing tokenizer settings.

#     Returns:
#         AutoTokenizer: A HuggingFace tokenizer instance.
#     """

#     if data_config.tokenizer.type == "local":
#         tokenizer = spm.SentencePieceProcessor(model_file=str(data_config.tokenizer.path_id))
#         return tokenizer

#     return AutoTokenizer.from_pretrained(data_config.tokenizer.name)

# def initialize_tokenizer(data_config: DataConfig):
#     """Initialize the tokenizer for text processing with SentencePiece support."""
    
#     tokenizer_name = data_config.tokenizer.path_id
    
#     # Check if this is a SentencePiece model path
#     if tokenizer_name.endswith('.model') or tokenizer_name.endswith('.sp'):
#         # This is a SentencePiece model file
#         return SentencePieceTokenizerWrapper(
#             model_path=tokenizer_name,
#             # You can customize these special tokens as needed
#             bos_token="<s>", 
#             eos_token="</s>",
#             unk_token="<unk>",
#             pad_token="<pad>"
#         )
#     else:
#         # Fall back to HuggingFace AutoTokenizer
#         return AutoTokenizer.from_pretrained(tokenizer_name)

def initialize_tokenizer(data_config):
    """
    Initialize tokenizer based on data config for training.
    
    This replaces the standard initialize_tokenizer function in your training utils.
    """
    tokenizer_name = data_config.tokenizer.path_id
    
    # Check if this is a path to a SentencePiece model
    if tokenizer_name.endswith('.model') or os.path.exists(tokenizer_name):
        return initialize_sentencepiece_tokenizer(
            model_path=tokenizer_name,
            vocab_size=getattr(data_config.tokenizer, 'vocab_size', None)
        )
    else:
        return AutoTokenizer.from_pretrained(tokenizer_name)


def initialize_dataloader(
    data_config: 'DataConfig',
    training_config: 'TrainingConfig',
    fabric: L.Fabric,
    train_dataset: 'Dataset',
    val_dataset: 'Dataset',
    tokenizer  # <-- 1. Add tokenizer_path as an argument
):
    """Initialize the DataLoader with on-the-fly tokenization and padding."""

    pad_id = tokenizer.pad_token_id # Get the padding token ID

    def _collate_fn(batch):
        """
        Processes a batch of raw text, tokenizes, and pads sequences.
        `batch` is a list of dictionaries, e.g., [{'text': '...'}, {'text': '...'}]
        """
        # 3. Extract the raw text from each sample in the batch
        # This assumes your dataset items are dictionaries with a 'text' key.
        # Adjust the key if your dataset uses a different one (e.g., 'sentence').
        texts = [entry['text'] for entry in batch]

        # 4. Tokenize each text string and convert to a PyTorch tensor
        # We add BOS/EOS tokens for good measure, as many models expect them.
        tokenized_batch = [
            torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id])
            for text in texts
        ]

        # 5. Pad all sequences in the batch to the same length
        # `pad_sequence` creates a rectangular tensor.
        padded_batch = pad_sequence(
            tokenized_batch, batch_first=True, padding_value=pad_id
        )

        # 6. Return a dictionary containing the single, padded tensor
        return {"input_ids": padded_batch}

    # The logic for calculating sub-batch size remains the same
    sub_batch_size = data_config.dataloader.batch_size // (
        fabric.world_size * training_config.optimization.gradient_accumulation_steps
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        # shuffle=True,  # Shuffle is generally recommended unless you have a reason not to
        pin_memory=True,
        collate_fn=_collate_fn, # Use our new, powerful collate function
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=sub_batch_size,
        # shuffle=False,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    return train_dataloader, val_dataloader

########################################################
#
# Model Initialization
#
########################################################

def initialize_model(model_config: ModelConfig):
    """Initialize a custom Pico model for training.
    
    Args:
        model_config: Configuration object containing custom model settings.
        
    Returns:
        PyTorch model instance (PicoDecoder).
    """
    if model_config.model_type == "pico_decoder":
        return PicoDecoder(model_config)
    elif model_config.model_type == "gpt2":
        return initialize_gpt2_model(model_config)
    elif model_config.model_type == "huggingface":
        return AutoModelForCausalLM.load(model_config["model_checkpoint"])
    else:
        raise ValueError(f"Invalid custom model type: {model_config.model_type}")
    
########################################################
#
# Optimizer and Scheduler
#
########################################################


def initialize_optimizer(training_config: TrainingConfig, model: torch.nn.Module):
    """Initialize the optimizer for model training.

    Creates an optimizer instance based on the configuration settings.

    Add whatever other optimizers you want here.

    Args:
        training_config: Configuration object containing optimizer settings.
            Must have:
            - optimization.optimizer (str): Name of the optimizer ("adamw")
            - optimization.lr (float): Learning rate for the optimizer
        model: PyTorch model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    """

    if training_config.optimization.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(training_config.optimization.lr)
        )
    else:
        raise ValueError(f"Invalid optimizer: {training_config.optimization.optimizer}")

    return optimizer


def initialize_lr_scheduler(
    training_config: TrainingConfig, optimizer: torch.optim.Optimizer
):
    """Initialize a learning rate scheduler with warmup and decay.

    The default is a learning rate scheduler that implements a linear warmup followed by
    linear decay. The learning rate increases linearly from 0 to the initial lr
    during warmup, then decreases linearly to 0 during the remaining steps.

    Add other types of learning rate schedulers here.

    Args:
        training_config: Configuration object containing optimizer and scheduler settings.
        optimizer: PyTorch optimizer whose learning rate will be scheduled.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler instance.
    """

    if training_config.optimization.lr_scheduler == "linear_with_warmup":
        # Credit where credit is due:
        # https://github.com/huggingface/transformers/blob/e71a01a104dd663c730e494eb0b6467bb51df357/src/transformers/optimization.py#L102
        def _lr_lambda(curr_step, num_warmup_steps, max_steps):
            if curr_step < num_warmup_steps:
                return float(curr_step) / float(max(1, num_warmup_steps))
            else:
                return max(
                    0.0,
                    float(max_steps - curr_step)
                    / float(max(1, max_steps - num_warmup_steps)),
                )

        lr_lambda = lambda step: _lr_lambda(  # noqa: E731
            step,
            training_config.optimization.lr_warmup_steps,
            training_config.max_steps,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
    else:
        raise ValueError(
            f"Invalid learning rate scheduler: {training_config.optimization.lr_scheduler}"
        )

    return lr_scheduler


########################################################
#
# Experiment Monitoring (Logging, Experiment Tracking, etc.)
#
########################################################


def _initialize_log_file(checkpointing_config: CheckpointingConfig) -> str:
    """Create and initialize a timestamped log file in the run's log directory.

    Sets up a log file with a unique timestamp in the run's logging directory.
    Creates the necessary directory structure if it doesn't exist.

    Directory Structure:
        {checkpointing_config.runs_dir}/
        └── {checkpointing_config.run_name}/
            └── {checkpointing_config.logs_dir}/
                └── log_YYYYMMDD_HHMMSS.txt

    Args:
        checkpointing_config: Configuration object containing checkpointing settings.

    Returns:
        str: Absolute path to the created log file.

    """

    run_dir = os.path.join(checkpointing_config.runs_dir, checkpointing_config.run_name)
    logs_dir = os.path.join(run_dir, checkpointing_config.logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    # datetime stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"log_{timestamp}.log"
    log_file_path = os.path.join(logs_dir, log_file_name)

    open(log_file_path, "w").close()  # Create an empty log file

    return log_file_path


@use_backoff()
def initialize_wandb(
    monitoring_config: MonitoringConfig, checkpointing_config: CheckpointingConfig
):
    """Initialize Weights and Biases.

    This function initializes Weights and Biases based on the configuration settings.

    Args:
        monitoring_config: Configuration object containing monitoring settings.
        checkpointing_config: Configuration object containing checkpointing settings.

    Returns:
        Optional[WandbLogger]: An experiment tracker instance.
    """

    assert (
        monitoring_config.wandb.project is not None
        and monitoring_config.wandb.project != ""
    ), "Wandb project must be provided if wandb is to be used."
    assert (
        monitoring_config.wandb.entity is not None
        and monitoring_config.wandb.entity != ""
    ), "Wandb entity must be provided if wandb is to be used."

    _run_id = None
    if checkpointing_config.training.auto_resume:
        # If we are loading a checkpoint, we can try to find the run id of the previous run
        previous_runs = wandb.Api().runs(
            path=f"{monitoring_config.wandb.entity}/{monitoring_config.wandb.project}",
            filters={"display_name": checkpointing_config.run_name},
        )
        try:
            if len(previous_runs) == 1:
                _run_id = previous_runs[0].id
        except ValueError:
            pass

    wandb_logger = WandbLogger(
        project=monitoring_config.wandb.project,
        entity=monitoring_config.wandb.entity,
        id=_run_id,
        name=checkpointing_config.run_name,
    )

    return wandb_logger


# @rank_zero_only
# def initialize_logging(
#     monitoring_config: MonitoringConfig,
#     checkpointing_config: CheckpointingConfig,
#     fabric: L.Fabric,
# ):
#     """Initialize logging system with default logging, to file and console.

#     The default logging system uses a file handler and a stream handler.

#     NOTE: this function is only called on rank 0.

#     Args:
#         monitoring_config: Configuration object containing monitoring settings.
#         checkpointing_config: Configuration object containing checkpointing settings.

#     Returns:
#         logger: Standard Python logger configured for file and console output
#     """

#     # ---- Standard Local Logger ---- #
#     logger = logging.getLogger("pico-train")
#     logger.setLevel(logging.INFO)

#     # Create file handler
#     log_file_path = _initialize_log_file(checkpointing_config)
#     file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
#     file_handler.setLevel(monitoring_config.logging.log_level)

#     # Create formatter and add it to the handler
#     formatter = logging.Formatter(
#         "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     file_handler.setFormatter(formatter)

#     # Add the handler to the logger
#     logger.addHandler(file_handler)

#     # Add a stream handler for console output
#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(monitoring_config.logging.log_level)
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)

#     return logger

@rank_zero_only
def initialize_logging(
    monitoring_config: MonitoringConfig,
    checkpointing_config: CheckpointingConfig,
    fabric: L.Fabric,
):
    """Initialize HPC-friendly logging system."""
    
    logger = logging.getLogger("pico-train")
    logger.setLevel(logging.INFO)
    
    # Use local temp directory for log file instead of shared storage
    log_dir = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"pico_train_{timestamp}.log")
    
    # Create file handler with buffering disabled for HPC
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(monitoring_config.logging.log_level)
    
    # Simple formatter to reduce overhead
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Use stdout instead of stderr and force flushing
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(monitoring_config.logging.log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Force immediate flushing for HPC environments
    for handler in logger.handlers:
        handler.flush()
    
    return logger


########################################################
#
# HuggingFace/Remote Checkpointing
#
########################################################


@rank_zero_only
@use_backoff()
def initialize_hf_checkpointing(
    checkpointing_config: CheckpointingConfig, fabric: L.Fabric
):
    """Initialize HuggingFace Checkpointing.

    Creates a HuggingFace repository if it doesn't exist, and creates a branch named after the run.

    NOTE: this function is only called on rank 0.

    Args:
        checkpointing_config: Configuration object containing checkpointing settings; must have
            a 'hf_checkpoint' attribute that specifies the HuggingFace repository id and
            collection slug (if applicable) to save the checkpoint to.

    Raises:
        RuntimeError: If unable to create HuggingFace repository after multiple attempts.
    """

    huggingface_repo_id = checkpointing_config.hf_checkpoint.repo_id
    assert (
        huggingface_repo_id is not None and huggingface_repo_id != ""
    ), "hf_checkpoint.repo_id must be provided."

    repo = create_repo(huggingface_repo_id, exist_ok=True)

    # can create a repo without a specified namespace (will default to username)
    # however the rest of the HF calls need the fully qualified name
    # this is returned by create repo, so we update the config for later calls
    checkpointing_config.hf_checkpoint.repo_id = repo.repo_id
    huggingface_repo_id = repo.repo_id

    if checkpointing_config.hf_checkpoint.collection_slug:
        add_collection_item(
            checkpointing_config.hf_checkpoint.collection_slug,
            huggingface_repo_id,
            repo.repo_type,
            exists_ok=True,
        )

    create_branch(
        repo_id=huggingface_repo_id,
        branch=checkpointing_config.run_name,
        exist_ok=True,
    )
