# !pip install torch loguru numpy transformers datasets

"""
Adaptive Multi-Model Training Framework

This module implements a novel training approach that trains multiple models simultaneously
and dynamically allocates computational resources (memory) based on learning performance.
Models that learn faster receive more resources, creating a competitive training environment.

Main components:
- ModelArena: Manages multiple models and their training processes
- AdaptiveTrainer: Handles resource allocation and performance tracking
- PerformanceMonitor: Tracks and evaluates model learning rates
"""

import os
import time
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    BertForMaskedLM,
    RobertaForMaskedLM,
    T5ForConditionalGeneration,
    get_scheduler
)
from datasets import load_dataset
from loguru import logger
import matplotlib.pyplot as plt
from torch.cuda import memory_allocated, max_memory_allocated, reset_peak_memory_stats

# Configure logger
logger.remove()
logger.add(
    "adaptive_training.log",
    rotation="100 MB",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(lambda msg: print(msg), level="INFO")

@dataclass
class ModelConfig:
    """Configuration for a model in the arena."""
    name: str
    model_type: str
    model_id: str
    initial_memory_share: float = 0.1  # Initial fraction of memory
    current_memory_share: float = field(default=0.1)
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    batch_size: Optional[int] = None  # Will be calculated based on memory share

class PerformanceTracker:
    """Tracks and analyzes model performance metrics during training."""

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of evaluations to use for calculating learning rate
        """
        self.loss_history: Dict[str, List[float]] = {}
        self.eval_scores: Dict[str, List[float]] = {}
        self.learning_rates: Dict[str, List[float]] = {}
        self.memory_shares: Dict[str, List[float]] = {}
        self.window_size = window_size
        logger.info(f"Initialized PerformanceTracker with window_size={window_size}")

    def add_loss(self, model_name: str, loss: float) -> None:
        """Add a training loss value for a model."""
        if model_name not in self.loss_history:
            self.loss_history[model_name] = []
        self.loss_history[model_name].append(loss)

    def add_eval_score(self, model_name: str, score: float) -> None:
        """Add an evaluation score for a model."""
        if model_name not in self.eval_scores:
            self.eval_scores[model_name] = []
        self.eval_scores[model_name].append(score)

    def add_memory_share(self, model_name: str, share: float) -> None:
        """Track memory allocation for a model."""
        if model_name not in self.memory_shares:
            self.memory_shares[model_name] = []
        self.memory_shares[model_name].append(share)

    def calculate_learning_rate(self, model_name: str) -> float:
        """
        Calculate the learning rate (improvement rate) of a model
        based on recent evaluation scores.

        Returns:
            float: Learning rate (higher is better)
        """
        if model_name not in self.eval_scores:
            logger.warning(f"No evaluation scores available for {model_name}")
            return 0.0

        scores = self.eval_scores[model_name]
        if len(scores) < 2:
            logger.info(f"Not enough data to calculate learning rate for {model_name}")
            return 0.0

        # Use the last window_size evaluations to calculate learning rate
        window = min(self.window_size, len(scores) - 1)
        recent_scores = scores[-window-1:]

        # Calculate average improvement in scores
        improvements = [recent_scores[i+1] - recent_scores[i] for i in range(len(recent_scores)-1)]
        avg_improvement = sum(improvements) / len(improvements)

        # Account for model's current performance level to avoid biasing toward poorly performing models
        # that have more room to improve
        normalized_improvement = avg_improvement / (1.0 + abs(recent_scores[-1]))

        # Store the calculated learning rate
        if model_name not in self.learning_rates:
            self.learning_rates[model_name] = []
        self.learning_rates[model_name].append(normalized_improvement)

        logger.info(f"Model {model_name} learning rate: {normalized_improvement:.6f}")
        return normalized_improvement

    def get_memory_allocation_weights(self) -> Dict[str, float]:
        """
        Calculate memory allocation weights based on learning rates.

        Returns:
            Dict[str, float]: Dictionary mapping model names to allocation weights
        """
        weights = {}

        # Get the latest learning rate for each model
        learning_rates = {name: rates[-1] if rates else 0.0
                          for name, rates in self.learning_rates.items()}

        # Handle the case where all learning rates are negative or zero
        if all(rate <= 0 for rate in learning_rates.values()):
            logger.warning("All models have negative or zero learning rates, using default allocation")
            # Default to equal allocation
            equal_share = 1.0 / len(learning_rates) if learning_rates else 1.0
            return {name: equal_share for name in learning_rates}

        # Convert negative learning rates to small positive values
        min_positive_rate = min((rate for rate in learning_rates.values() if rate > 0), default=1e-6)
        adjusted_rates = {name: max(rate, min_positive_rate * 0.1)
                          for name, rate in learning_rates.items()}

        # Calculate weights proportional to learning rates
        total = sum(adjusted_rates.values())
        weights = {name: rate / total for name, rate in adjusted_rates.items()}

        # Apply smoothing to avoid extreme allocation changes
        current_shares = {name: shares[-1] if shares else 1.0 / len(learning_rates)
                          for name, shares in self.memory_shares.items()}

        smoothed_weights = {}
        smoothing_factor = 0.7  # How much to consider the current allocation

        for name in weights:
            smoothed_weights[name] = (
                smoothing_factor * current_shares.get(name, 0) +
                (1 - smoothing_factor) * weights[name]
            )

        # Renormalize to ensure weights sum to 1
        total_smoothed = sum(smoothed_weights.values())
        normalized_weights = {name: weight / total_smoothed
                              for name, weight in smoothed_weights.items()}

        # Ensure minimum allocation for all models
        min_allocation = 0.05
        for name in normalized_weights:
            if normalized_weights[name] < min_allocation:
                normalized_weights[name] = min_allocation

        # Renormalize again
        total = sum(normalized_weights.values())
        final_weights = {name: weight / total for name, weight in normalized_weights.items()}

        # Log the allocation decisions
        log_data = {name: f"{weight:.4f}" for name, weight in final_weights.items()}
        logger.info(f"Memory allocation weights: {json.dumps(log_data)}")

        return final_weights

    def plot_performance(self, output_dir: str = "./plots") -> None:
        """
        Generate performance visualizations.

        Args:
            output_dir: Directory to save plot images
        """
        os.makedirs(output_dir, exist_ok=True)

        # Plot loss history
        plt.figure(figsize=(12, 6))
        for model_name, losses in self.loss_history.items():
            if losses:
                plt.plot(losses, label=model_name)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "loss_history.png"))
        plt.close()

        # Plot evaluation scores
        plt.figure(figsize=(12, 6))
        for model_name, scores in self.eval_scores.items():
            if scores:
                plt.plot(scores, label=model_name)
        plt.xlabel("Evaluation Steps")
        plt.ylabel("Evaluation Score")
        plt.title("Model Performance Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "eval_scores.png"))
        plt.close()

        # Plot memory allocation
        plt.figure(figsize=(12, 6))
        for model_name, shares in self.memory_shares.items():
            if shares:
                plt.plot(shares, label=model_name)
        plt.xlabel("Allocation Steps")
        plt.ylabel("Memory Share")
        plt.title("Memory Allocation Over Time")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, "memory_allocation.png"))
        plt.close()

        # Plot learning rates
        plt.figure(figsize=(12, 6))
        for model_name, rates in self.learning_rates.items():
            if rates:
                plt.plot(rates, label=model_name)
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("Model Learning Rates Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "learning_rates.png"))
        plt.close()

        logger.info(f"Performance plots saved to {output_dir}")

class MemoryManager:
    """Manages memory allocation and usage tracking for multiple models."""

    def __init__(self, total_memory_gb: float = None):
        """
        Args:
            total_memory_gb: Total GPU memory in GB to allocate across models.
                             If None, will be estimated from available GPU memory.
        """
        # If total_memory_gb is not provided, estimate from available GPU memory
        if total_memory_gb is None:
            if torch.cuda.is_available():
                # Get total GPU memory and use 90% of it
                total_memory = torch.cuda.get_device_properties(0).total_memory
                self.total_memory = int(total_memory * 0.9)  # 90% of total memory in bytes
                total_memory_gb = self.total_memory / (1024**3)
                logger.info(f"Detected {total_memory_gb:.2f} GB GPU memory, using 90% for training")
            else:
                # Default value for CPU training
                self.total_memory = 8 * (1024**3)  # 8 GB in bytes
                total_memory_gb = 8
                logger.info(f"No GPU detected, using default {total_memory_gb} GB memory limit")
        else:
            self.total_memory = int(total_memory_gb * (1024**3))  # Convert GB to bytes

        self.model_memory_usage: Dict[str, int] = {}
        self.model_batch_sizes: Dict[str, int] = {}
        logger.info(f"Initialized MemoryManager with {total_memory_gb:.2f} GB total memory")

    def estimate_model_memory(self, model: nn.Module, sample_input_size: Tuple[int, int]) -> int:
        """
        Estimate memory usage for a model with a given input size.

        Args:
            model: PyTorch model
            sample_input_size: Tuple of (batch_size, sequence_length)

        Returns:
            int: Estimated memory usage in bytes
        """
        if torch.cuda.is_available():
            # Clear existing memory stats
            torch.cuda.empty_cache()
            reset_peak_memory_stats()

            # Move model to GPU if not already
            if next(model.parameters()).device.type != 'cuda':
                model = model.cuda()

            # Generate random input
            batch_size, seq_len = sample_input_size
            sample_input = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
            attention_mask = torch.ones((batch_size, seq_len), device='cuda')

            # Forward pass to measure memory
            with torch.no_grad():
                _ = model(input_ids=sample_input, attention_mask=attention_mask)

            # Get peak memory usage
            memory_used = max_memory_allocated()

            # Estimate memory for gradient storage and optimizer states (roughly 4x forward pass)
            total_estimate = memory_used * 5

            logger.info(f"Estimated memory for batch size {batch_size}: {total_estimate / (1024**2):.2f} MB")
            return total_estimate
        else:
            # Rough estimation for CPU mode based on model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            batch_size, seq_len = sample_input_size

            # Rough multiplier for activations, gradients, and optimizer states
            multiplier = 12
            memory_estimate = model_size * multiplier

            # Scale by batch size
            memory_estimate *= batch_size

            logger.info(f"Estimated CPU memory: {memory_estimate / (1024**2):.2f} MB")
            return memory_estimate

    def calculate_batch_size(self, model_name: str, model: nn.Module,
                             memory_share: float, sequence_length: int,
                             min_batch_size: int = 1, max_batch_size: int = 64) -> int:
        """
        Calculate optimal batch size for a model based on its memory allocation.

        Args:
            model_name: Name of the model
            model: PyTorch model
            memory_share: Fraction of total memory allocated to this model
            sequence_length: Input sequence length
            min_batch_size: Minimum batch size to allow
            max_batch_size: Maximum batch size to allow

        Returns:
            int: Optimal batch size
        """
        available_memory = self.total_memory * memory_share
        logger.info(f"Memory available for {model_name}: {available_memory / (1024**2):.2f} MB")

        # Binary search to find optimal batch size
        low, high = min_batch_size, max_batch_size
        optimal_batch_size = min_batch_size

        while low <= high:
            mid = (low + high) // 2
            memory_required = self.estimate_model_memory(model, (mid, sequence_length))

            if memory_required <= available_memory:
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1

        # Store the batch size
        self.model_batch_sizes[model_name] = optimal_batch_size

        logger.info(f"Optimal batch size for {model_name}: {optimal_batch_size}")
        return optimal_batch_size

class ModelArena:
    """
    Manages multiple models competing for resources during training.
    Implements the adaptive memory allocation strategy.
    """

    def __init__(
        self,
        model_configs: List[ModelConfig],
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        max_sequence_length: int = 128,
        evaluation_interval: int = 100,
        memory_reallocation_interval: int = 200,
        eval_steps: int = 20,
    ):
        """
        Args:
            model_configs: List of model configurations to train
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            max_sequence_length: Maximum sequence length for tokenization
            evaluation_interval: Steps between evaluations
            memory_reallocation_interval: Steps between memory reallocations
            eval_steps: Number of steps to use for evaluation
        """
        self.model_configs = model_configs
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_sequence_length = max_sequence_length
        self.evaluation_interval = evaluation_interval
        self.memory_reallocation_interval = memory_reallocation_interval
        self.eval_steps = eval_steps

        # Initialize components
        self.performance_tracker = PerformanceTracker()
        self.memory_manager = MemoryManager()

        # Training tracking variables
        self.global_step = 0
        self.model_steps = {config.name: 0 for config in model_configs}

        # Attributes to be initialized in setup
        self.models = {}
        self.tokenizers = {}
        self.optimizers = {}
        self.schedulers = {}
        self.train_dataloaders = {}
        self.eval_dataloaders = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialized ModelArena with {len(model_configs)} models")
        logger.info(f"Using device: {self.device}")
        for config in model_configs:
            logger.info(f"Model: {config.name}, Type: {config.model_type}, ID: {config.model_id}")

    def setup(self):
        """Initialize and prepare all models, tokenizers, and datasets."""
        logger.info("Setting up models and datasets...")

        # Load dataset
        self._load_dataset()

        # Initialize models
        for config in self.model_configs:
            self._initialize_model(config)

        # Initial memory allocation
        self._allocate_memory()

        logger.info("Setup complete!")

    def _load_dataset(self):
        """Load and prepare the dataset."""
        logger.info(f"Loading dataset {self.dataset_name}/{self.dataset_config}")

        # Load raw dataset
        dataset = load_dataset(self.dataset_name, self.dataset_config)
        self.raw_dataset = dataset

        logger.info(f"Dataset loaded: {len(dataset['train'])} training examples")
        logger.info(f"Validation set: {len(dataset['validation'])} examples")

    def _initialize_model(self, config: ModelConfig):
        """Initialize a model and its components based on its configuration."""
        logger.info(f"Initializing model {config.name} ({config.model_type})")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizers[config.name] = tokenizer

        # Load model based on type
        if config.model_type == "gpt2":
            model = GPT2LMHeadModel.from_pretrained(config.model_id)
        elif config.model_type == "bert":
            model = BertForMaskedLM.from_pretrained(config.model_id)
        elif config.model_type == "roberta":
            model = RobertaForMaskedLM.from_pretrained(config.model_id)
        elif config.model_type == "t5":
            model = T5ForConditionalGeneration.from_pretrained(config.model_id)
        else:
            # Default to auto model detection
            model = AutoModelForCausalLM.from_pretrained(config.model_id)

        model.to(self.device)
        self.models[config.name] = model

        # Initialize optimizer
        self.optimizers[config.name] = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )

        # Prepare tokenized datasets
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt"
            )

        # Tokenize datasets
        tokenized_train = self.raw_dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        tokenized_eval = self.raw_dataset["validation"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        # Set format to return PyTorch tensors directly
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Create dataloaders (batch size will be set later)
        # Initially use a minimal batch size of 1
        self.train_dataloaders[config.name] = DataLoader(
            tokenized_train,
            batch_size=1,
            shuffle=True
        )

        self.eval_dataloaders[config.name] = DataLoader(
            tokenized_eval,
            batch_size=1,
            shuffle=False
        )

        # Initialize scheduler
        self.schedulers[config.name] = get_scheduler(
            "linear",
            optimizer=self.optimizers[config.name],
            num_warmup_steps=config.warmup_steps,
            num_training_steps=10000  # Will be updated during training
        )

        logger.info(f"Model {config.name} initialized successfully")

    def _allocate_memory(self):
        """Allocate memory to models and adjust batch sizes."""
        logger.info("Allocating memory to models...")

        # Get allocation weights
        if any(self.performance_tracker.learning_rates.values()):
            # We have learning rate data, use it for allocation
            weights = self.performance_tracker.get_memory_allocation_weights()
        else:
            # Initial allocation based on configs
            weights = {config.name: config.initial_memory_share for config in self.model_configs}
            # Normalize to ensure they sum to 1
            total = sum(weights.values())
            weights = {name: weight / total for name, weight in weights.items()}

        # Update model configs with new memory shares
        for config in self.model_configs:
            config.current_memory_share = weights[config.name]
            self.performance_tracker.add_memory_share(config.name, config.current_memory_share)

        # Calculate batch sizes based on memory allocation
        for config in self.model_configs:
            model = self.models[config.name]
            batch_size = self.memory_manager.calculate_batch_size(
                config.name,
                model,
                config.current_memory_share,
                self.max_sequence_length
            )

            # Update config
            config.batch_size = batch_size

            # Update dataloaders with new batch size
            tokenized_train = self.train_dataloaders[config.name].dataset
            tokenized_eval = self.eval_dataloaders[config.name].dataset

            self.train_dataloaders[config.name] = DataLoader(
                tokenized_train,
                batch_size=batch_size,
                shuffle=True
            )

            self.eval_dataloaders[config.name] = DataLoader(
                tokenized_eval,
                batch_size=batch_size,
                shuffle=False
            )

            logger.info(f"Model {config.name}: Memory share = {config.current_memory_share:.4f}, "
                       f"Batch size = {batch_size}")

    def _evaluate_model(self, model_name: str) -> float:
        """
        Evaluate a model on the validation set.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            float: Evaluation score (lower is better for perplexity)
        """
        logger.info(f"Evaluating model {model_name}")

        model = self.models[model_name]
        eval_dataloader = self.eval_dataloaders[model_name]

        model.eval()
        total_loss = 0
        steps = 0

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= self.eval_steps:
                    break

                # Check batch format and handle accordingly
                if isinstance(batch, dict):
                    # Dictionary format - expected format
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # For models like BERT/RoBERTa that need labels for masked tokens
                    if model_name.startswith(("bert", "roberta")):
                        batch["labels"] = batch["input_ids"].clone()
                    else:
                        # For autoregressive models like GPT-2
                        batch["labels"] = batch["input_ids"].clone()

                    # Forward pass
                    outputs = model(**batch)

                elif isinstance(batch, (list, tuple)):
                    # List/tuple format - convert to dictionary
                    # Assume first element is input_ids, second is attention_mask if available
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

                    # Prepare inputs
                    model_inputs = {"input_ids": input_ids}
                    if attention_mask is not None:
                        model_inputs["attention_mask"] = attention_mask

                    # For models like BERT/RoBERTa that need labels for masked tokens
                    if model_name.startswith(("bert", "roberta")):
                        model_inputs["labels"] = input_ids.clone()
                    else:
                        # For autoregressive models like GPT-2
                        model_inputs["labels"] = input_ids.clone()

                    # Forward pass
                    outputs = model(**model_inputs)
                else:
                    logger.error(f"Unexpected batch format: {type(batch)}")
                    continue

                loss = outputs.loss
                total_loss += loss.item()
                steps += 1

        # Calculate average loss
        avg_loss = total_loss / steps if steps > 0 else float('inf')

        # For language models, we use negative loss as score (higher is better)
        # This makes it consistent with other metrics where higher is better
        score = -avg_loss

        # Add to performance tracker
        self.performance_tracker.add_eval_score(model_name, score)

        logger.info(f"Model {model_name} evaluation: Loss = {avg_loss:.4f}, Score = {score:.4f}")
        return score

    def train(self, total_steps: int = 5000, save_dir: str = "./model_outputs"):
        """
        Train all models with adaptive memory allocation.

        Args:
            total_steps: Total number of training steps
            save_dir: Directory to save model checkpoints and plots
        """
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Starting training for {total_steps} steps")

        # Reset training state
        self.global_step = 0
        self.model_steps = {config.name: 0 for config in self.model_configs}
        model_iterators = {config.name: iter(self.train_dataloaders[config.name])
                          for config in self.model_configs}

        # Initial evaluation
        logger.info("Performing initial evaluation")
        for config in self.model_configs:
            self._evaluate_model(config.name)

        # Calculate initial learning rates
        for config in self.model_configs:
            self.performance_tracker.calculate_learning_rate(config.name)

        # Initial memory allocation based on configs
        self._allocate_memory()

        # Main training loop
        while self.global_step < total_steps:
            # Train each model for a few steps
            for config in self.model_configs:
                model_name = config.name
                model = self.models[model_name]
                optimizer = self.optimizers[model_name]
                scheduler = self.schedulers[model_name]

                # Get steps proportional to memory allocation
                steps_to_train = max(1, int(config.current_memory_share * 10))

                # Train the model
                model.train()
                accumulated_loss = 0

                for _ in range(steps_to_train):
                    # Get the next batch, restart iterator if needed
                    try:
                        batch = next(model_iterators[model_name])
                    except StopIteration:
                        model_iterators[model_name] = iter(self.train_dataloaders[model_name])
                        batch = next(model_iterators[model_name])

                    # Handle different batch formats
                    if isinstance(batch, dict):
                        # Dictionary format - expected format
                        # Move batch to device
                        batch = {k: v.to(self.device) for k, v in batch.items()}

                        # For models like BERT/RoBERTa that need labels for masked tokens
                        if model_name.startswith(("bert", "roberta")):
                            batch["labels"] = batch["input_ids"].clone()
                        else:
                            # For autoregressive models like GPT-2
                            batch["labels"] = batch["input_ids"].clone()

                    elif isinstance(batch, (list, tuple)):
                        # List/tuple format - convert to dictionary
                        # Assume first element is input_ids, second is attention_mask if available
                        input_ids = batch[0].to(self.device)
                        attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

                        # Prepare inputs
                        batch = {"input_ids": input_ids}
                        if attention_mask is not None:
                            batch["attention_mask"] = attention_mask

                        # For models like BERT/RoBERTa that need labels for masked tokens
                        if model_name.startswith(("bert", "roberta")):
                            batch["labels"] = input_ids.clone()
                        else:
                            # For autoregressive models like GPT-2
                            batch["labels"] = input_ids.clone()
                    else:
                        logger.error(f"Unexpected batch format: {type(batch)}")
                        continue

                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
                    accumulated_loss += loss.item()

                    # Backward pass
                    loss.backward()

                    # Update weights if enough gradients are accumulated
                    if (self.model_steps[model_name] + 1) % config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    self.model_steps[model_name] += 1

                # Log training progress
                avg_loss = accumulated_loss / steps_to_train
                self.performance_tracker.add_loss(model_name, avg_loss)
                logger.info(f"Model {model_name} step {self.model_steps[model_name]}: "
                           f"Loss = {avg_loss:.4f}, "
                           f"Memory share = {config.current_memory_share:.4f}")

            # Increment global step
            self.global_step += 1

            # Periodic evaluation
            if self.global_step % self.evaluation_interval == 0:
                logger.info(f"Global step {self.global_step}: Evaluating all models")
                for config in self.model_configs:
                    self._evaluate_model(config.name)
                    self.performance_tracker.calculate_learning_rate(config.name)

                # Generate performance plots
                self.performance_tracker.plot_performance(
                    output_dir=os.path.join(save_dir, "plots")
                )

            # Periodic memory reallocation
            if self.global_step % self.memory_reallocation_interval == 0:
                logger.info(f"Global step {self.global_step}: Reallocating memory")
                self._allocate_memory()

            # Save checkpoints periodically
            if self.global_step % 1000 == 0:
                self._save_checkpoints(save_dir, self.global_step)

        # Final evaluation and memory allocation
        logger.info("Final evaluation")
        for config in self.model_configs:
            self._evaluate_model(config.name)

        # Calculate final learning rates
        for config in self.model_configs:
            self.performance_tracker.calculate_learning_rate(config.name)

        # Final memory allocation
        self._allocate_memory()

        # Save final models and plots
        self._save_checkpoints(save_dir, self.global_step)
        self.performance_tracker.plot_performance(
            output_dir=os.path.join(save_dir, "plots")
        )

        # Determine the winner
        winner = self._determine_winner()
        logger.info(f"Training complete! Winner: {winner}")

        return winner

    def _save_checkpoints(self, save_dir: str, step: int):
        """Save model checkpoints and training state."""
        checkpoint_dir = os.path.join(save_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save each model
        for config in self.model_configs:
            model_name = config.name
            model_dir = os.path.join(checkpoint_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            self.models[model_name].save_pretrained(model_dir)
            self.tokenizers[model_name].save_pretrained(model_dir)

            # Save optimizer state
            torch.save(
                self.optimizers[model_name].state_dict(),
                os.path.join(model_dir, "optimizer.pt")
            )

            # Save scheduler state
            torch.save(
                self.schedulers[model_name].state_dict(),
                os.path.join(model_dir, "scheduler.pt")
            )

        # Save training state and configuration
        training_state = {
            "global_step": step,
            "model_steps": {config.name: self.model_steps[config.name] for config in self.model_configs},
            "model_configs": [vars(config) for config in self.model_configs],
            "performance_tracker": {
                "loss_history": self.performance_tracker.loss_history,
                "eval_scores": self.performance_tracker.eval_scores,
                "learning_rates": self.performance_tracker.learning_rates,
                "memory_shares": self.performance_tracker.memory_shares
            }
        }

        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)

        logger.info(f"Saved checkpoints at step {step} to {checkpoint_dir}")

    def _determine_winner(self) -> str:
        """
        Determine the winning model based on final evaluation scores.

        Returns:
            str: Name of the winning model
        """
        # Get the last evaluation score for each model
        final_scores = {
            name: scores[-1] if scores else float('-inf')
            for name, scores in self.performance_tracker.eval_scores.items()
        }

        # Find the model with the highest score
        winner = max(final_scores.items(), key=lambda x: x[1])[0]

        # Log the final rankings
        rankings = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info("Final model rankings:")
        for rank, (name, score) in enumerate(rankings, 1):
            logger.info(f"{rank}. {name}: Score = {score:.4f}")

        return winner


# def run_adaptive_training_experiment(
#     output_dir: str = "./adaptive_training_results",
#     total_steps: int = 5000,
#     dataset_name: str = "wikitext",
#     dataset_config: str = "wikitext-2-raw-v1",
#     max_sequence_length: int = 128,
#     evaluation_interval: int = 100,
#     memory_reallocation_interval: int = 200
# ) -> str:
#     """
#     Run an adaptive training experiment with multiple models.

#     Args:
#         output_dir: Directory to save results
#         total_steps: Total number of training steps
#         dataset_name: HuggingFace dataset name
#         dataset_config: Dataset configuration
#         max_sequence_length: Maximum sequence length for tokenization
#         evaluation_interval: Steps between evaluations
#         memory_reallocation_interval: Steps between memory reallocations

#     Returns:
#         str: Name of the winning model
#     """
#     # Configure logging for this run
#     os.makedirs(output_dir, exist_ok=True)
#     log_file = os.path.join(output_dir, "experiment.log")
#     logger.add(log_file, rotation="100 MB")

#     logger.info("=" * 80)
#     logger.info("Starting adaptive training experiment")
#     logger.info(f"Output directory: {output_dir}")
#     logger.info(f"Dataset: {dataset_name}/{dataset_config}")
#     logger.info(f"Total steps: {total_steps}")
#     logger.info("=" * 80)

#     # Define model configurations
#     model_configs = [
#         ModelConfig(
#             name="gpt2-small",
#             model_type="gpt2",
#             model_id="gpt2",
#             initial_memory_share=0.25,
#             learning_rate=5e-5,
#             gradient_accumulation_steps=2
#         ),
#         ModelConfig(
#             name="gpt2-medium",
#             model_type="gpt2",
#             model_id="gpt2-medium",
#             initial_memory_share=0.25,
#             learning_rate=4e-5,
#             gradient_accumulation_steps=4
#         ),
#         ModelConfig(
#             name="bert-base",
#             model_type="bert",
#             model_id="bert-base-uncased",
#             initial_memory_share=0.25,
#             learning_rate=3e-5,
#             gradient_accumulation_steps=1
#         ),
#         ModelConfig(
#             name="roberta-base",
#             model_type="roberta",
#             model_id="roberta-base",
#             initial_memory_share=0.25,
#             learning_rate=3e-5,
#             gradient_accumulation_steps=1
#         )
#     ]

#     # Initialize the arena
#     arena = ModelArena(
#         model_configs=model_configs,
#         dataset_name=dataset_name,
#         dataset_config=dataset_config,
#         max_sequence_length=max_sequence_length,
#         evaluation_interval=evaluation_interval,
#         memory_reallocation_interval=memory_reallocation_interval
#     )

#     # Setup models and datasets
#     arena.setup()

#     # Start training
#     winner = arena.train(
#         total_steps=total_steps,
#         save_dir=output_dir
#     )

#     logger.info(f"Experiment completed. Winning model: {winner}")
#     logger.info(f"Results saved to {output_dir}")

#     return winner


# # Example usage
# if __name__ == "__main__":
#     # Smaller dataset for quicker experimentation
#     winner = run_adaptive_training_experiment(
#         total_steps=1000,
#         dataset_name="wikitext",
#         dataset_config="wikitext-2-raw-v1",
#         max_sequence_length=128,
#         evaluation_interval=50,
#         memory_reallocation_interval=100
#     )

#     print(f"Winning model: {winner}")


# """
# Adaptive Multi-Model Training Framework Extension for Modern 3B LLMs

# This experiment extends the original Adaptive Multi-Model Training Framework to use
# more modern 3B parameter range LLMs such as Llama, Mistral, and other similar models.
# """

# import os
# import torch
# from typing import List
# from loguru import logger

# # Import from the original framework
# from transformers import AutoModelForCausalLM, AutoTokenizer


def run_modern_llm_experiment(
    output_dir: str = "./modern_llm_results",
    total_steps: int = 2000,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_sequence_length: int = 512,
    evaluation_interval: int = 100,
    memory_reallocation_interval: int = 200,
    use_peft: bool = True,
):
    """
    Run an adaptive training experiment with modern 3B parameter LLMs.
    
    Args:
        output_dir: Directory to save results
        total_steps: Total number of training steps
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        max_sequence_length: Maximum sequence length for tokenization
        evaluation_interval: Steps between evaluations
        memory_reallocation_interval: Steps between memory reallocations
        use_peft: Whether to use Parameter-Efficient Fine-Tuning (PEFT)
        
    Returns:
        str: Name of the winning model
    """
    # Configure logging for this run
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "experiment.log")
    logger.add(log_file, rotation="100 MB")
    
    logger.info("=" * 80)
    logger.info("Starting modern LLM adaptive training experiment")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {dataset_name}/{dataset_config}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Using PEFT: {use_peft}")
    logger.info("=" * 80)
    
    # Check GPU availability and warn if resources might be insufficient
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Available GPU memory: {gpu_memory_gb:.2f} GB")
        if gpu_memory_gb < 24:
            logger.warning("Warning: 3B parameter models may require 24+ GB of GPU memory.")
            logger.warning("Consider using PEFT methods or smaller models if OOM errors occur.")
    else:
        logger.warning("No GPU detected! Training 3B parameter models on CPU is not recommended.")
    
    # Define model configurations for modern 3B LLMs
    model_configs = [
        ModelConfig(
            name="llama-3b",
            model_type="llama",
            model_id="meta-llama/Llama-3.2-3B-Instruct",  # We'll load with low_cpu_mem_usage and device_map="auto"
            initial_memory_share=0.33,
            learning_rate=2e-5,
            gradient_accumulation_steps=8,
        ),
        ModelConfig(
            name="mistral-3b",
            model_type="mistral",
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",  # We'll load with low_cpu_mem_usage and device_map="auto"
            initial_memory_share=0.33,
            learning_rate=2e-5,
            gradient_accumulation_steps=8,
        ),
        ModelConfig(
            name="falcon-3b",
            model_type="falcon",
            model_id="bigscience/bloomz-3b",  # We'll load with low_cpu_mem_usage and device_map="auto"
            initial_memory_share=0.33,
            learning_rate=2e-5,
            gradient_accumulation_steps=8,
        ),
    ]
    
    # Extend the ModelArena class to handle modern LLMs
    class ModernLLMArena(ModelArena):
        def _initialize_model(self, config: ModelConfig):
            """Override model initialization to handle modern LLMs and apply PEFT if needed."""
            logger.info(f"Initializing model {config.name} ({config.model_type})")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizers[config.name] = tokenizer

            # Load model with optimizations for large models
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            }
            
            if torch.cuda.is_available():
                # Use device_map="auto" for automatic model sharding across GPUs if multiple are available
                model_kwargs["device_map"] = "auto"
            
            # Load model
            logger.info(f"Loading {config.name} with optimizations...")
            
            try:
                # For all model types, try using AutoModelForCausalLM first
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id, 
                    **model_kwargs
                )
                
                # Check if we have enough memory to load the full model
                if torch.cuda.is_available():
                    available_memory = torch.cuda.get_device_properties(0).total_memory
                    required_memory = model.get_memory_footprint() * 2  # Approximate memory needed during training
                    
                    if required_memory > available_memory:
                        logger.warning(f"Model {config.name} requires more memory than available.")
                        logger.warning(f"Required: {required_memory / 1e9:.2f} GB, Available: {available_memory / 1e9:.2f} GB")
                        logger.warning("Consider enabling PEFT or using smaller models.")
                
                # Apply PEFT if requested
                if use_peft:
                    try:
                        from peft import get_peft_model, LoraConfig, TaskType
                        
                        logger.info(f"Applying LoRA to {config.name}")
                        
                        # Configure LoRA
                        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            inference_mode=False,
                            r=16,  # rank
                            lora_alpha=32,
                            lora_dropout=0.1,
                            target_modules=["q_proj", "v_proj"]  # Appropriate target modules for most transformer models
                        )
                        
                        # Apply PEFT configuration
                        model = get_peft_model(model, peft_config)
                        logger.info(f"LoRA applied successfully to {config.name}")
                        
                        # Log number of trainable parameters
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        total_params = sum(p.numel() for p in model.parameters())
                        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
                        
                    except ImportError:
                        logger.warning("PEFT library not found. Installing with pip...")
                        import subprocess
                        subprocess.run(["pip", "install", "peft"], check=True)
                        
                        # Try again after installation
                        from peft import get_peft_model, LoraConfig, TaskType
                        
                        logger.info(f"Applying LoRA to {config.name}")
                        
                        # Configure LoRA
                        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            inference_mode=False,
                            r=16,
                            lora_alpha=32,
                            lora_dropout=0.1,
                            target_modules=["q_proj", "v_proj"]
                        )
                        
                        # Apply PEFT configuration
                        model = get_peft_model(model, peft_config)
                        logger.info(f"LoRA applied successfully to {config.name}")
                    
                    except Exception as e:
                        logger.error(f"Failed to apply PEFT to {config.name}: {str(e)}")
                        logger.warning("Continuing without PEFT...")
            
            except Exception as e:
                logger.error(f"Error loading model {config.name}: {str(e)}")
                logger.error("Attempting to load a smaller version or alternative model...")
                
                # Fallback to smaller models if original fails
                fallback_models = {
                    "llama-3b": "facebook/opt-1.3b",
                    "mistral-3b": "EleutherAI/pythia-1.4b",
                    "falcon-3b": "EleutherAI/gpt-neo-1.3B",
                    "phi-3b": "microsoft/phi-1_5"
                }
                
                if config.name in fallback_models:
                    fallback_id = fallback_models[config.name]
                    logger.info(f"Falling back to {fallback_id}")
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        fallback_id,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                else:
                    # If no specific fallback, use a very small model
                    logger.info("Falling back to a small GPT-2 model")
                    model = AutoModelForCausalLM.from_pretrained("gpt2")
            
            # Move model to device (if not already handled by device_map="auto")
            if not use_peft and model.device.type != "cuda" and torch.cuda.is_available():
                logger.info(f"Moving {config.name} to GPU...")
                model.to(self.device)
                
            self.models[config.name] = model
            
            # Continue with the rest of the initialization as in the original method
            # (optimizer, dataloaders, etc.)
            super()._initialize_model(config)
            
        def _evaluate_model(self, model_name: str) -> float:
            """Override evaluation to handle modern LLMs' specific features."""
            # Check if the model uses PEFT
            model = self.models[model_name]
            is_peft_model = hasattr(model, 'base_model')
            
            # For PEFT models, we might need special handling during evaluation
            if is_peft_model and use_peft:
                logger.info(f"Evaluating PEFT model {model_name}")
                # Special PEFT evaluation logic can be added here if needed
            
            # Call the original evaluation method
            return super()._evaluate_model(model_name)
    
    # Initialize the ModernLLMArena
    arena = ModernLLMArena(
        model_configs=model_configs,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_sequence_length=max_sequence_length,
        evaluation_interval=evaluation_interval,
        memory_reallocation_interval=memory_reallocation_interval
    )
    
    # Setup models and datasets
    try:
        arena.setup()
        
        # Start training
        winner = arena.train(
            total_steps=total_steps,
            save_dir=output_dir
        )
        
        logger.info(f"Experiment completed. Winning model: {winner}")
        logger.info(f"Results saved to {output_dir}")
        
        return winner
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Consider using PEFT, reducing the number of models, or using smaller models.")
        return None


if __name__ == "__main__":
    # You may need to install additional libraries for this experiment
    # pip install peft accelerate bitsandbytes
    
    # Run the experiment with Parameter-Efficient Fine-Tuning
    winner = run_modern_llm_experiment(
        total_steps=500,  # Reduced for initial testing
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_sequence_length=512,
        evaluation_interval=50,
        memory_reallocation_interval=100,
        use_peft=True,  # Enable PEFT for memory efficiency
    )
    
    print(f"Winning model: {winner}")
