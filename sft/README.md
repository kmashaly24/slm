# Supervised fine-tuning

This phase turns the finance-pretrained Qwen model into an English analyst for Egyptian stocks. The training answers were written by Gemini from a hidden future price path. That path is never shown to the model as part of the user message.

The finished artifact of this phase is `sft/merged_16bit`. GRPO starts from that folder.

## How the data was created

`creating_sft_data.py` writes `sft_dataset.jsonl`. The file used for training has 3,120 examples. The script is resumable: rows already on disk are skipped, and a Gemini quota error stops the run so it can be continued after a new key is added.

For every ticker and date the script:

1. Downloads daily Yahoo bars with `market_data.py`. The chart endpoint is used first because the installed `yfinance` crumb request comes back empty.
2. Picks target dates from 6 June 2023 up to today minus 12 days, spaced four trading sessions apart, at most 200 dates per ticker. Each date needs 60 past sessions and 5 future closes.
3. Summarizes the 60 sessions strictly before the date: start close, last close, percent change, high, low, and the last 10 closes.
4. Fetches news from seven calendar days before the date through the date with `news_fetch.py`.
5. Asks Gemini (`gemini-2.5-flash`, through `gemini_client.py`) to write the assistant answer. The hidden prompt includes the realized T+1..T+5 closes so the written reasoning matches that path. The saved user message does not include those closes.

News comes from Google News RSS, English, Egypt edition. Each example has three blocks, at most two headlines each:

- **Corporate.** The headline must name that company. Short ambiguous names also need an Egypt marker.
- **Sector.** Egyptian industry headlines for banking, real estate, industrials, energy, or consumer. The company name is not required.
- **Macro.** Egypt inflation, rates, the pound, the IMF, and foreign shocks that reach Egypt (Fed, dollar, oil price or supply, wheat, Gulf, Suez, Red Sea). The same macro block is shared by every ticker on the same dates.

A date is kept when corporate or macro has at least one real headline. An empty block is written as `none in this window.` Arabic-heavy blocks are translated to English before they are saved.

The primary tickers are CIB, EFG Hermes, Fawry, Beltone, Talaat Moustafa, Palm Hills, Madinet Masr, Heliopolis Housing, Elsewedy, Ezz Steel, Oriental Weavers, Abu Qir, AMOC, Sidi Kerir, Eastern Company, Juhayna, Ibnsina Pharma, Telecom Egypt, Alexandria Containers, and Qalaa. Backup names in `config.py` are used only if the primary list cannot fill 4,000 examples. The saved file stopped at 3,120.

Each row is one chat:

- system: Egyptian analyst, English, no invented numbers, do not mention future prices
- user: ticker, company, date, 60-session summary, news, and one task
- assistant: the Gemini answer

The three tasks, defined in `prompts.py`, are:

- **A.** Predict the likely direction over the next five trading days and explain the reasoning.
- **B.** Identify the main risks and classify near-term sentiment as bullish or bearish.
- **C.** Explain how macro and sector news is likely to affect this stock.

Which task a date receives is chosen with a seeded random draw, so a rerun assigns the same task to the same ticker and date.

## Training design

`merge_phase1.py` merges the Phase 1 LoRA into a fresh `Qwen/Qwen2.5-1.5B` download and writes `mid_training/phase1/merged_16bit`. SFT loads that merged model in 4-bit and trains a new LoRA. It does not continue the Phase 1 adapter in place.

`train_sft.py` uses TRL `SFTTrainer` and Unsloth `train_on_responses_only`, so the loss is on the assistant tokens only. Unsloth is imported before Transformers. TensorFlow is disabled with `USE_TF=0`. Hugging Face transfer is off, and the download timeout is 120 seconds.

| Setting | Value |
| --- | --- |
| Base | `mid_training/phase1/merged_16bit` |
| Method | Unsloth QLoRA, 4-bit load, bfloat16 compute |
| LoRA | `r=32`, `alpha=64`, dropout 0, same seven target modules as Phase 1 |
| Sequence length | 2,048 |
| Epochs | 3 |
| Learning rate | `2e-4`, cosine, warmup ratio 0.05, weight decay 0.01 |
| Batch | 1 per device, gradient accumulation 16 |
| Optimizer | `adamw_8bit` |
| Validation | 10% of examples |
| Log / eval / save | every 20 / 100 / 100 steps |
| Seed | 42 |

Three epochs were the full supervised run. More epochs were not added. The best validation checkpoint is `checkpoints/checkpoint-500`, with validation loss 0.7725. `final_adapter/` is that adapter. `merged_16bit/` is the adapter merged into 16-bit weights.

## How to run it

From the repository root, in the CUDA environment. Do not reinstall Unsloth on Windows if it is already installed.

```powershell
pip install -r sft/requirements.txt
```

Put Gemini keys in `.env` as `GEMINI_API_KEYS`, using `.env.example` as the shape. Do not commit that file.

```powershell
python -m sft.merge_phase1
python -m sft.creating_sft_data
python -m sft.train_sft
python -m sft.train_sft --resume
python -m sft.train_sft --export-gguf
```

`creating_sft_data` can be run again after a quota stop. `--resume` continues SFT from the latest checkpoint. `--export-gguf` writes a `q4_k_m` GGUF under `sft/gguf/` after the adapter exists. The next phase needs `sft/merged_16bit`, which the trainer exports at the end of a finished run.
