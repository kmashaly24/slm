"""Dr. GRPO alignment of the Phase 2 EGX model on a 4 GB GPU.

Loads sft/merged_16bit in 4-bit and trains a fresh LoRA with TRL's GRPOTrainer.
Loss is Dr. GRPO: global-constant token normalization and no reward std scaling.

    python -m grpo.train_grpo
    python -m grpo.train_grpo --resume
    python -m grpo.train_grpo --export-gguf
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
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import RepeatSampler

from grpo.config import (
    BF16,
    FP16,
    GRPO_BETA,
    GRPO_CHECKPOINT_DIR,
    GRPO_DATASET_PATH,
    GRPO_EVAL_NUM_GENERATIONS,
    GRPO_EVAL_STEPS,
    GRPO_FINAL_ADAPTER_DIR,
    GRPO_GGUF_DIR,
    GRPO_GRAD_ACCUM,
    GRPO_LEARNING_RATE,
    GRPO_LOGGING_STEPS,
    GRPO_LORA_ALPHA,
    GRPO_LORA_DROPOUT,
    GRPO_LORA_R,
    GRPO_LR_SCHEDULER,
    GRPO_MAX_COMPLETION_LENGTH,
    GRPO_MAX_PROMPT_LENGTH,
    GRPO_MAX_STEPS,
    GRPO_MERGED_16BIT_DIR,
    GRPO_NUM_GENERATIONS,
    GRPO_PER_DEVICE_EVAL_BATCH_SIZE,
    GRPO_PER_DEVICE_TRAIN_BATCH_SIZE,
    GRPO_SAVE_STEPS,
    GRPO_VAL_FRACTION,
    GRPO_SEED,
    GRPO_TEMPERATURE,
    GRPO_WARMUP_RATIO,
    GRPO_WEIGHT_DECAY,
    GGUF_QUANT,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    OPTIM,
    SFT_MERGED_DIR,
)
from grpo.grpo_dataset_gen import split_by_recent_dates
from grpo.rewards import (
    accuracy_reward_func,
    format_reward_func,
    think_length,
    xml_count_reward_func,
)


def print_parameter_report(model) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    print("\n=== Parameter Report ===")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({pct:.4f}%)")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def _prompt_row(row: dict) -> dict:
    return {"prompt": row["prompt"], "target_answer": row["target_answer"]}


def load_prompt_datasets() -> tuple[Dataset, Dataset]:
    rows = []
    with open(GRPO_DATASET_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(
            f"No prompts in {GRPO_DATASET_PATH}. Run `python -m grpo.grpo_dataset_gen` first."
        )
    print(f"GRPO prompts: {len(rows):,}. Holding out the most recent {GRPO_VAL_FRACTION:.0%} of dates per ticker.")
    train_rows, val_rows = split_by_recent_dates(rows, GRPO_VAL_FRACTION)
    if not val_rows:
        raise RuntimeError("Validation split is empty. Need at least two dates per ticker.")
    print(f"Train prompts: {len(train_rows):,}  Validation prompts: {len(val_rows):,}")
    return Dataset.from_list([_prompt_row(row) for row in train_rows]), Dataset.from_list(
        [_prompt_row(row) for row in val_rows]
    )


def load_model():
    if not (SFT_MERGED_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Missing Phase 2 merged model at {SFT_MERGED_DIR}. Run Phase 2 training first."
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(SFT_MERGED_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=GRPO_LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=GRPO_LORA_ALPHA,
        lora_dropout=GRPO_LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=GRPO_SEED,
    )
    print_parameter_report(model)
    return model, tokenizer


def _training_args() -> GRPOConfig:
    kwargs = dict(
        output_dir=str(GRPO_CHECKPOINT_DIR),
        loss_type="dr_grpo",
        scale_rewards=False,
        beta=GRPO_BETA,
        num_generations=GRPO_NUM_GENERATIONS,
        temperature=GRPO_TEMPERATURE,
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH,
        max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
        max_steps=GRPO_MAX_STEPS,
        learning_rate=GRPO_LEARNING_RATE,
        lr_scheduler_type=GRPO_LR_SCHEDULER,
        warmup_ratio=GRPO_WARMUP_RATIO,
        weight_decay=GRPO_WEIGHT_DECAY,
        per_device_train_batch_size=GRPO_PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=GRPO_PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRPO_GRAD_ACCUM,
        gradient_checkpointing=True,
        optim=OPTIM,
        fp16=FP16 and torch.cuda.is_available(),
        bf16=BF16 and torch.cuda.is_available(),
        logging_steps=GRPO_LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=GRPO_EVAL_STEPS,
        save_strategy="steps",
        save_steps=GRPO_SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_reward",
        greater_is_better=True,
        seed=GRPO_SEED,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        use_vllm=False,
    )
    fields = GRPOConfig.__dataclass_fields__
    strategy_key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
    filtered = {key: value for key, value in kwargs.items() if key in fields and key != "eval_strategy"}
    return GRPOConfig(**{strategy_key: kwargs["eval_strategy"], **filtered})


class MonitoredGRPOTrainer(GRPOTrainer):
    """GRPO trainer that records <think> length and scores one validation answer."""

    def _get_eval_sampler(self, eval_dataset):
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=GRPO_EVAL_NUM_GENERATIONS,
            batch_size=1,
            repeat_count=1,
            shuffle=False,
            seed=self.args.seed,
        )

    def _generate_and_score_completions(self, inputs):
        # Captured before scoring. The model can leave eval mode during that work,
        # and think length must stay on the same side of the log as the rewards.
        self._metric_mode = "train" if self.model.training else "eval"
        if self._metric_mode == "train" or GRPO_EVAL_NUM_GENERATIONS == self.num_generations:
            return super()._generate_and_score_completions(inputs)
        saved = self.num_generations
        self.num_generations = GRPO_EVAL_NUM_GENERATIONS
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            self.num_generations = saved

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        self._record_think_lengths(completions)
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    def _record_think_lengths(self, completions) -> None:
        if not completions:
            return
        mode = getattr(self, "_metric_mode", None)
        if mode not in ("train", "eval"):
            mode = "train" if self.model.training else "eval"
        lengths = [think_length(completion, self.processing_class) for completion in completions]
        self._metrics[mode]["completions/mean_think_length"].append(sum(lengths) / len(lengths))

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        snapshot = {
            key: sum(values) / len(values)
            for key, values in self._metrics[mode].items()
            if values
        }
        # Keep these on the caller's dict. The best-checkpoint check reads that
        # object, and the parent logger copies metrics into a new dict.
        prefix = "eval_" if mode == "eval" else ""
        for key, value in snapshot.items():
            logs[f"{prefix}{key}"] = value
        super().log(logs, start_time)
        _print_monitor(mode, self.state.global_step, logs, snapshot)


def _print_monitor(mode: str, step: int, logs: dict, snapshot: dict) -> None:
    loss_key = "loss" if mode == "train" else "eval_loss"
    fields = (
        ("loss", logs.get(loss_key)),
        ("reward", snapshot.get("reward")),
        ("format", snapshot.get("rewards/format_reward_func/mean")),
        ("xml", snapshot.get("rewards/xml_count_reward_func/mean")),
        ("accuracy", snapshot.get("rewards/accuracy_reward_func/mean")),
        ("completion_len", snapshot.get("completions/mean_length")),
        ("think_len", snapshot.get("completions/mean_think_length")),
    )
    parts = [f"[{mode}] step {step}"]
    for name, value in fields:
        if name in ("completion_len", "think_len"):
            rendered = "n/a" if value is None else f"{value:.1f}"
        else:
            rendered = "n/a" if value is None else f"{value:.4f}"
        parts.append(f"{name} {rendered}")
    print(" | ".join(parts), flush=True)


def export_models(model, tokenizer, export_gguf: bool) -> None:
    GRPO_FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(GRPO_FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(GRPO_FINAL_ADAPTER_DIR))
    print(f"Saved GRPO adapter to {GRPO_FINAL_ADAPTER_DIR}")

    GRPO_MERGED_16BIT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(GRPO_MERGED_16BIT_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Saved merged 16-bit model to {GRPO_MERGED_16BIT_DIR}")

    if not export_gguf:
        print("Skipping GGUF export. Pass --export-gguf to build a q4_k_m file.")
        return
    GRPO_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting GGUF. This downloads llama.cpp helpers and can take a while on Windows.")
    model.save_pretrained_gguf(
        str(GRPO_GGUF_DIR),
        tokenizer,
        quantization_method=GGUF_QUANT,
    )
    print(f"Saved GGUF to {GRPO_GGUF_DIR}")


def train(resume: bool, export_gguf: bool) -> None:
    if not GRPO_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {GRPO_DATASET_PATH}. Run `python -m grpo.grpo_dataset_gen` first."
        )
    model, tokenizer = load_model()
    train_dataset, eval_dataset = load_prompt_datasets()
    trainer = MonitoredGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, xml_count_reward_func, accuracy_reward_func],
        args=_training_args(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=resume)
    export_models(model, tokenizer, export_gguf=export_gguf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 EGX Dr. GRPO")
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
