"""Phase 3 paths, sampling rules, return bins, and 4 GB GRPO hyperparameters."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = Path(__file__).resolve().parent

# Phase 2 merged weights. Loaded in 4-bit; a fresh GRPO LoRA is attached on top.
SFT_MERGED_DIR = ROOT / "sft" / "merged_16bit"

GRPO_DATASET_PATH = GRPO_DIR / "grpo_dataset.jsonl"
GRPO_CHECKPOINT_DIR = GRPO_DIR / "checkpoints"
GRPO_FINAL_ADAPTER_DIR = GRPO_DIR / "final_adapter"
GRPO_MERGED_16BIT_DIR = GRPO_DIR / "merged_16bit"
GRPO_GGUF_DIR = GRPO_DIR / "gguf"

GRPO_SAMPLES_PER_TICKER = 150
GRPO_HORIZONS = (1, 3, 5, 10)
GRPO_MAX_HORIZON = 10
GRPO_START = "2024-01-01"
GRPO_SEED = 123

# Exact answer strings. Thresholds are applied in grpo_dataset_gen.label_for_return.
GRPO_LABELS = (
    "Strong Bullish",
    "Slight Bullish",
    "Neutral",
    "Slight Bearish",
    "Strong Bearish",
)

# Same modules as Phase 2, copied here so sft/config.py stays unchanged.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
MAX_SEQ_LENGTH = 2048

GRPO_LORA_R = 16
GRPO_LORA_ALPHA = 32
GRPO_LORA_DROPOUT = 0.0

GRPO_NUM_GENERATIONS = 4
GRPO_LEARNING_RATE = 5e-6
GRPO_WARMUP_RATIO = 0.05
GRPO_WEIGHT_DECAY = 0.01
GRPO_LR_SCHEDULER = "cosine"
GRPO_MAX_PROMPT_LENGTH = 1024
GRPO_MAX_COMPLETION_LENGTH = 512
GRPO_GRAD_ACCUM = 8
GRPO_MAX_STEPS = 600
GRPO_LOGGING_STEPS = 10
GRPO_SAVE_STEPS = 75
GRPO_EVAL_STEPS = 75
GRPO_VAL_FRACTION = 0.15
# One answer per validation prompt. Training still samples GRPO_NUM_GENERATIONS.
GRPO_EVAL_NUM_GENERATIONS = 1
# Must be a multiple of GRPO_NUM_GENERATIONS. Each validation step scores this many prompts.
GRPO_PER_DEVICE_EVAL_BATCH_SIZE = 4
GRPO_BETA = 0.0
GRPO_TEMPERATURE = 1.0
# One backward pass. Accumulation 8 generates 8 answers per update
# (2 prompts x 4). Raising this batch multiplies that generation.
GRPO_PER_DEVICE_TRAIN_BATCH_SIZE = 1
OPTIM = "adamw_8bit"
FP16 = False
BF16 = True
GGUF_QUANT = "q4_k_m"
