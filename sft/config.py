"""Phase 2 paths, EGX universe, news mix, and QLoRA hyperparameters."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = Path(__file__).resolve().parent

PHASE1_ADAPTER_DIR = ROOT / "mid_training" / "phase1" / "final_adapter"
MERGED_BASE_DIR = ROOT / "mid_training" / "phase1" / "merged_16bit"

DATASET_PATH = SFT_DIR / "sft_dataset.jsonl"
CHECKPOINT_DIR = SFT_DIR / "checkpoints"
FINAL_ADAPTER_DIR = SFT_DIR / "final_adapter"
MERGED_16BIT_DIR = SFT_DIR / "merged_16bit"
GGUF_DIR = SFT_DIR / "gguf"

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_SEQ_LENGTH = 2048
TARGET_EXAMPLES = 4000
MAX_DATES_PER_TICKER = 200
DATE_SPACING_TRADING_DAYS = 4
LOOKBACK_CANDLES = 60
NEWS_LOOKBACK_DAYS = 7
FUTURE_DAYS = 5
# First target date.
EARLIEST_START = "2023-06-06"
# Leave room after T so T+1..T+5 closes exist.
CUTOFF_LAG_DAYS = 12

SEED = 42
TASKS = ("A", "B", "C")

# At most two headlines in each block. A date is kept when corporate or macro
# has at least one real headline. Sector is allowed to be empty.
NEWS_MIX = {"corporate": 2, "sector": 2, "macro": 2}

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_SLEEP_SECONDS = 4.0
GEMINI_MAX_RETRIES = 4

# ticker -> (arabic name, english name, sector key)
TICKER_META: dict[str, tuple[str, str, str]] = {
    "COMI.CA": ("البنك التجاري الدولي", "CIB", "banking"),
    "HRHO.CA": ("المجموعة المالية هيرميس", "EFG Hermes", "banking"),
    "FWRY.CA": ("فوري", "Fawry", "banking"),
    "BTFH.CA": ("بلتون المالية القابضة", "Beltone", "banking"),
    "TMGH.CA": ("مجموعة طلعت مصطفى", "Talaat Moustafa", "real_estate"),
    "PHDC.CA": ("بالم هيلز للتعمير", "Palm Hills", "real_estate"),
    "MASR.CA": ("مدينة مصر للإسكان", "Madinet Masr", "real_estate"),
    "ORAS.CA": ("أوراسكوم كونستراكشون", "Orascom Construction", "real_estate"),
    "SWDY.CA": ("السويدي إليكتريك", "Elsewedy Electric", "industrials"),
    "ESRS.CA": ("حديد عز", "Ezz Steel", "industrials"),
    "ORWE.CA": ("النساجون الشرقيون", "Oriental Weavers", "industrials"),
    "ABUK.CA": ("أبو قير للأسمدة", "Abu Qir Fertilizers", "energy"),
    "AMOC.CA": ("أموك الإسكندرية للزيوت المعدنية", "AMOC", "energy"),
    "SKPC.CA": ("سيدي كرير للبتروكيماويات", "Sidi Kerir", "energy"),
    "EAST.CA": ("الشرقية للدخان", "Eastern Company", "consumer"),
    "JUFO.CA": ("جهينة", "Juhayna", "consumer"),
    "ISPH.CA": ("ابن سينا فارما", "Ibnsina Pharma", "consumer"),
    "ETEL.CA": ("المصرية للاتصالات", "Telecom Egypt", "consumer"),
    "ALCN.CA": ("الإسكندرية لتداول الحاويات", "Alexandria Containers", "consumer"),
    "CCAP.CA": ("القلعة القابضة", "Qalaa Holdings", "consumer"),
    "EKHO.CA": ("القابضة المصرية الكويتية", "EK Holding", "energy"),
    "CIEB.CA": ("كريدي أجريكول مصر", "Credit Agricole Egypt", "banking"),
    "MFPC.CA": ("موبكو مصر لإنتاج الأسمدة", "MOPCO", "energy"),
    "SUGR.CA": ("الدلتا للسكر", "Delta Sugar", "consumer"),
    "OLFI.CA": ("عبور لاند", "Obour Land", "consumer"),
    "EFID.CA": ("إيديتا", "Edita", "consumer"),
    "GBCO.CA": ("جي بي أوتو", "GB Auto", "industrials"),
    "MTIE.CA": ("إم إم جروب", "MM Group", "consumer"),
    "EFIH.CA": ("إي فاينانس", "e-finance", "banking"),
    "ADIB.CA": ("مصرف أبوظبي الإسلامي مصر", "ADIB Egypt", "banking"),
    "HELI.CA": ("مصر الجديدة للإسكان", "Heliopolis Housing", "real_estate"),
    "MNHD.CA": ("مدينة نصر للإسكان", "MNHD", "real_estate"),
    "SVCE.CA": ("جنوب الوادي للأسمنت", "South Valley Cement", "industrials"),
    "ORHD.CA": ("أوراسكوم للتنمية مصر", "Orascom Development", "real_estate"),
}

PRIMARY_TICKERS = [
    "COMI.CA", "HRHO.CA", "FWRY.CA", "BTFH.CA",
    "TMGH.CA", "PHDC.CA", "MASR.CA", "HELI.CA",
    "SWDY.CA", "ESRS.CA", "ORWE.CA",
    "ABUK.CA", "AMOC.CA", "SKPC.CA",
    "EAST.CA", "JUFO.CA", "ISPH.CA", "ETEL.CA", "ALCN.CA", "CCAP.CA",
]

BACKUP_TICKERS = [
    "EKHO.CA", "CIEB.CA", "MFPC.CA", "SUGR.CA", "OLFI.CA", "EFID.CA",
    "GBCO.CA", "MTIE.CA", "EFIH.CA", "ADIB.CA", "MNHD.CA",
    "SVCE.CA", "ORHD.CA",
]

# Industry words for the sector block. Rates, the pound, and inflation stay in macro.
SECTOR_KEYWORDS = {
    "banking": (
        "Egyptian banks",
        "Egypt banks",
        "banks in Egypt",
        "banks operating in Egypt",
        "banking sector",
    ),
    "real_estate": ("real estate", "housing", "construction", "developer", "developers"),
    "industrials": ("steel", "cement", "electricity", "manufacturing"),
    "energy": ("oil", "gas", "fertilizer", "fertilizers", "petrochemical", "fuel"),
    "consumer": ("consumer", "food", "medicine", "pharma", "telecom", "retail"),
}

CORPORATE_SITES = "site:mubasher.info OR site:alborsaanews.com"
SECTOR_SITES = "site:hapijournal.com OR site:alborsaanews.com"
MACRO_SITES = "site:enterprise.press OR site:asharqbusiness.com"
# Egypt-scoped search. The word "oil" stays in the query so the date limits
# still work. Price and supply shocks are enforced on the headline itself.
MACRO_QUERY = (
    'Egypt (inflation OR "interest rate" OR "central bank" OR pound OR IMF '
    'OR "Federal Reserve" OR dollar OR oil OR wheat OR Suez OR "Red Sea" '
    "OR remittances OR Gulf)"
)

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]

NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LOGGING_STEPS = 20
EVAL_STEPS = 100
SAVE_STEPS = 100
VAL_FRACTION = 0.10
OPTIM = "adamw_8bit"
FP16 = False
BF16 = True
GGUF_QUANT = "q4_k_m"
