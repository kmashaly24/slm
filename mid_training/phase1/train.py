"""QLoRA continued pre-training on the packed Phase 1 dataset.

Run from the repository root:

    python -m phase1.train
    python -m phase1.train --resume
"""

from __future__ import annotations

import argparse
import math
import os

# This Anaconda env has TensorFlow plus protobuf 4.25.9. TensorFlow cannot
# import (it needs google.protobuf.runtime_version), and transformers pulls
# TensorFlow in whenever it is installed. Training uses PyTorch only.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

import unsloth  # noqa: F401  -- must run before transformers/peft
from unsloth import FastLanguageModel

import torch
from datasets import load_from_disk
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator

from phase1.config import (
    BF16,
    CHECKPOINT_DIR,
    EVAL_STEPS,
    FINAL_ADAPTER_DIR,
    FP16,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LR_SCHEDULER,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    OPTIM,
    PACKED_DATASET_DIR,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    SEED,
    VAL_FRACTION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def print_parameter_report(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    print("\n=== Parameter Report ===")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({pct:.4f}%)")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    print_parameter_report(model)
    return model, tokenizer


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss = logs["loss"]
        if loss is None:
            return
        try:
            logs["perplexity"] = math.exp(min(loss, 20))
        except OverflowError:
            logs["perplexity"] = float("inf")
        lr = logs.get("learning_rate")
        print(
            f"step={state.global_step} loss={loss:.4f} "
            f"perplexity={logs['perplexity']:.4f} lr={lr}"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        loss = metrics["eval_loss"]
        metrics["eval_perplexity"] = math.exp(min(loss, 20))
        print(
            f"eval step={state.global_step} loss={loss:.4f} "
            f"perplexity={metrics['eval_perplexity']:.4f}"
        )


def train(resume: bool) -> None:
    if not PACKED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Missing {PACKED_DATASET_DIR}. Run `python -m phase1.build_dataset` first."
        )
    model, tokenizer = load_model()
    dataset = load_from_disk(str(PACKED_DATASET_DIR))
    split = dataset.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    print(
        f"Train sequences: {len(split['train']):,}  "
        f"Validation sequences: {len(split['test']):,}"
    )

    # Sequences are already packed to MAX_SEQ_LENGTH, so this is plain causal-LM
    # training rather than a second SFT tokenization pass.
    arg_kwargs = dict(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim=OPTIM,
        fp16=FP16 and torch.cuda.is_available(),
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    try:
        args = TrainingArguments(eval_strategy="steps", **arg_kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="steps", **arg_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=default_data_collator,
        callbacks=[PerplexityCallback()],
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    print(f"Saved best adapter to {FINAL_ADAPTER_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 QLoRA mid-training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in phase1/checkpoints.",
    )
    args = parser.parse_args()
    train(resume=args.resume)


if __name__ == "__main__":
    main()
