"""QLoRA SFT on the Gemini-distilled EGX dataset.

Response tokens are the only ones that contribute to the loss. The base is
the merged Phase 1 checkpoint, loaded in 4-bit for a 4 GB RTX 3050.

    python -m sft.train_sft
    python -m sft.train_sft --resume
    python -m sft.train_sft --export-gguf
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

import unsloth  # noqa: F401
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from sft.config import (
    BF16,
    CHECKPOINT_DIR,
    DATASET_PATH,
    EVAL_STEPS,
    FINAL_ADAPTER_DIR,
    FP16,
    GGUF_DIR,
    GGUF_QUANT,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LR_SCHEDULER,
    MAX_SEQ_LENGTH,
    MERGED_16BIT_DIR,
    MERGED_BASE_DIR,
    NUM_EPOCHS,
    OPTIM,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    SEED,
    VAL_FRACTION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from sft.merge_phase1 import merge


def print_parameter_report(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    print("\n=== Parameter Report ===")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({pct:.4f}%)")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def load_chat_dataset(tokenizer) -> Dataset:
    rows = []
    with open(DATASET_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            rows.append({"text": text})
    if not rows:
        raise FileNotFoundError(f"No examples in {DATASET_PATH}. Run creating_sft_data first.")
    return Dataset.from_list(rows)


def load_model():
    if not (MERGED_BASE_DIR / "config.json").exists():
        merge()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MERGED_BASE_DIR),
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


def _training_args() -> TrainingArguments:
    kwargs = dict(
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
        bf16=BF16 and torch.cuda.is_available(),
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to="none",
        dataloader_pin_memory=False,
        dataset_text_field="text",
        packing=False,
    )
    from trl import SFTConfig

    fields = SFTConfig.__dataclass_fields__
    strategy_key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
    filtered = {key: value for key, value in kwargs.items() if key in fields}
    return SFTConfig(**{strategy_key: "steps", **filtered})


def export_models(model, tokenizer, export_gguf: bool) -> None:
    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    print(f"Saved best adapter to {FINAL_ADAPTER_DIR}")

    MERGED_16BIT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED_16BIT_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Saved merged 16-bit model to {MERGED_16BIT_DIR}")

    if not export_gguf:
        print("Skipping GGUF export. Pass --export-gguf to build a q4_k_m file.")
        return
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting GGUF. This downloads llama.cpp helpers and can take a while on Windows.")
    model.save_pretrained_gguf(
        str(GGUF_DIR),
        tokenizer,
        quantization_method=GGUF_QUANT,
    )
    print(f"Saved GGUF to {GGUF_DIR}")


def train(resume: bool, export_gguf: bool) -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}. Run `python -m sft.creating_sft_data` first."
        )
    model, tokenizer = load_model()
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    dataset = load_chat_dataset(tokenizer)
    split = dataset.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    print(
        f"Train examples: {len(split['train']):,}  "
        f"Validation examples: {len(split['test']):,}"
    )
    args = _training_args()
    trainer_kwargs = dict(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=args,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,
            **trainer_kwargs,
        )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train(resume_from_checkpoint=resume)
    export_models(model, tokenizer, export_gguf=export_gguf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 EGX QLoRA SFT")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Also write a q4_k_m GGUF. Optional; llama.cpp tooling can be slow on Windows.",
    )
    args = parser.parse_args()
    train(resume=args.resume, export_gguf=args.export_gguf)


if __name__ == "__main__":
    main()
