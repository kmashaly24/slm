from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "mid_training_data"
BOOKS_DIR = DATA_ROOT / "books"
ARTICLES_DIR = DATA_ROOT / "articles"
CORPUS_DIR = DATA_ROOT / "corpus"
PAGES_JSON = BOOKS_DIR / "pages.json"

PHASE1_DIR = Path(__file__).resolve().parent
PACKED_DATASET_DIR = PHASE1_DIR / "packed_dataset"
CHECKPOINT_DIR = PHASE1_DIR / "checkpoints"
FINAL_ADAPTER_DIR = PHASE1_DIR / "final_adapter"
TOKEN_REPORT_PATH = PHASE1_DIR / "token_report.json"

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_SEQ_LENGTH = 2048
TARGET_TOKENS = 30_000_000

SEED = 42
ARTICLE_SAMPLE_FRACTION = 0.10
MIN_ARTICLE_WORDS = 300
MIN_BOOK_BLOCK_WORDS = 20
TABLE_DIGIT_SYMBOL_RATIO = 0.30
MIN_CORPUS_SECTION_WORDS = 40

# Narrative 10-K items only: Business, Risk Factors, Properties, Legal, MD&A.
CORPUS_SECTIONS = (
    "section_1",
    "section_1A",
    "section_2",
    "section_3",
    "section_7",
    "section_7A",
)

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
LOGGING_STEPS = 20
EVAL_STEPS = 40
SAVE_STEPS = 40
VAL_FRACTION = 0.05
OPTIM = "adamw_8bit"
# RTX 3050 supports bfloat16. Unsloth's fast LoRA kernel crashes under fp16
# with "self and mat2 must have the same dtype, but got Half and Float".
FP16 = False
BF16 = True
