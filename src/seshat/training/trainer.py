"""QLoRA fine-tuning trainer for Seshat.

Wraps Hugging Face Transformers + PEFT + TRL to provide a simple interface
for fine-tuning multilingual LLMs on new languages.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class SeshatTrainingConfig:
    """Consolidated training configuration."""

    # Model
    base_model: str = "Qwen/Qwen3-8B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    packing: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    dataloader_num_workers: int = 2

    # Output
    output_dir: str = "outputs/training"
    merge_and_save: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path, phase: int | None = None, **overrides) -> "SeshatTrainingConfig":
        """Load config from YAML file with optional phase overrides.

        Args:
            config_path: Path to base YAML config.
            phase: Training phase (1-4) to apply phase-specific overrides.
            **overrides: Additional overrides (e.g., model name from CLI).
        """
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # Flatten the nested YAML structure into flat kwargs
        kwargs: dict = {}

        # Model section
        model_cfg = raw.get("model", {})
        kwargs["base_model"] = model_cfg.get("base_model", cls.base_model)
        kwargs["torch_dtype"] = model_cfg.get("torch_dtype", cls.torch_dtype)
        kwargs["attn_implementation"] = model_cfg.get("attn_implementation", cls.attn_implementation)
        kwargs["trust_remote_code"] = model_cfg.get("trust_remote_code", cls.trust_remote_code)

        # Quantization section
        quant_cfg = raw.get("quantization", {})
        kwargs["load_in_4bit"] = quant_cfg.get("load_in_4bit", cls.load_in_4bit)
        kwargs["bnb_4bit_quant_type"] = quant_cfg.get("bnb_4bit_quant_type", cls.bnb_4bit_quant_type)
        kwargs["bnb_4bit_compute_dtype"] = quant_cfg.get("bnb_4bit_compute_dtype", cls.bnb_4bit_compute_dtype)
        kwargs["bnb_4bit_use_double_quant"] = quant_cfg.get("bnb_4bit_use_double_quant", cls.bnb_4bit_use_double_quant)

        # LoRA section
        lora_cfg = raw.get("lora", {})
        kwargs["lora_r"] = lora_cfg.get("r", cls.lora_r)
        kwargs["lora_alpha"] = lora_cfg.get("lora_alpha", cls.lora_alpha)
        kwargs["lora_dropout"] = lora_cfg.get("lora_dropout", cls.lora_dropout)
        kwargs["lora_target_modules"] = lora_cfg.get("target_modules", cls.lora_target_modules.__func__())

        # Training section
        train_cfg = raw.get("training", {})
        for key in [
            "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate",
            "lr_scheduler_type", "warmup_ratio", "weight_decay",
            "max_grad_norm", "max_seq_length", "packing",
            "gradient_checkpointing", "optim", "bf16",
            "logging_steps", "save_steps", "eval_steps",
            "save_total_limit", "seed", "dataloader_num_workers",
        ]:
            if key in train_cfg:
                kwargs[key] = train_cfg[key]

        # Output section
        output_cfg = raw.get("output", {})
        kwargs["output_dir"] = output_cfg.get("output_dir", cls.output_dir)
        kwargs["merge_and_save"] = output_cfg.get("merge_and_save", cls.merge_and_save)

        # Apply phase-specific overrides
        if phase is not None:
            phase_key = {1: "1_vocabulary", 2: "2_translation", 3: "3_analysis", 4: "4_generation"}.get(phase)
            phases_cfg = raw.get("phases", {})
            if phase_key and phase_key in phases_cfg:
                phase_overrides = phases_cfg[phase_key]
                kwargs.update(phase_overrides)
                logger.info("Applied phase %d overrides: %s", phase, phase_overrides)

        # Apply CLI overrides last
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

        return cls(**kwargs)


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def load_model_and_tokenizer(
    config: SeshatTrainingConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model with quantization and prepare for LoRA training.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, tokenizer) ready for LoRA adapter attachment.
    """
    logger.info("Loading model: %s", config.base_model)

    compute_dtype = _get_torch_dtype(config.bnb_4bit_compute_dtype)

    # Quantization config
    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        torch_dtype=_get_torch_dtype(config.torch_dtype),
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )

    # Prepare for k-bit training (enables gradient computation on quantized model)
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(
        "Model loaded. Parameters: %.1fB, Quantized: %s",
        sum(p.numel() for p in model.parameters()) / 1e9,
        config.load_in_4bit,
    )
    return model, tokenizer


def create_lora_config(config: SeshatTrainingConfig) -> LoraConfig:
    """Create LoRA configuration from training config."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train(
    config: SeshatTrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    resume_from_checkpoint: bool = False,
) -> Path:
    """Run QLoRA fine-tuning.

    Args:
        config: Training configuration.
        train_dataset: Training dataset with 'messages' column.
        eval_dataset: Optional evaluation dataset.
        resume_from_checkpoint: Whether to resume from latest checkpoint.

    Returns:
        Path to the output directory containing the trained adapter.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Create LoRA config
    lora_config = create_lora_config(config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        bf16=config.bf16,
        fp16=False,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to="none",  # Set to "wandb" or "tensorboard" if desired
        remove_unused_columns=False,
    )

    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        max_seq_length=config.max_seq_length,
        packing=config.packing,
    )

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.2f%%)",
        trainable_params,
        total_params,
        100 * trainable_params / total_params,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the adapter
    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Adapter saved to %s", adapter_path)

    # Optionally merge LoRA into base model
    if config.merge_and_save:
        logger.info("Merging LoRA adapter into base model...")
        merged_model = trainer.model.merge_and_unload()
        merged_path = output_dir / "merged_model"
        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        logger.info("Merged model saved to %s", merged_path)

    return output_dir
