# Dr. GRPO

This phase takes the supervised EGX model and trains it to name a five-way return class. The public weights are the best checkpoint from that run:

[karimmashaly/egx-qwen2.5-1.5b-grpo](https://huggingface.co/karimmashaly/egx-qwen2.5-1.5b-grpo)

The same merged 16-bit folder is `grpo/merged_16bit`. The local web app loads that folder.

## How the data was created

`grpo_dataset_gen.py` writes `grpo_dataset.jsonl`. The file has 2,850 rows: 19 tickers times 150 dates. ESRS is missing because Yahoo had no usable history. No Gemini calls are made. Price and news reuse `sft/market_data.py` and `sft/news_fetch.py`.

Sampling rules in `config.py`:

- Dates run from 1 January 2024 through the sample end.
- The end is the same cutoff as supervised training: today minus 12 days.
- A date is kept only when it has 60 past bars and 10 future closes.
- The horizon `H` is one of 1, 3, 5, or 10 trading days.
- Every sampled date is kept even when the news block is empty.
- The seed is 123, and each ticker contributes 150 rows.

The prompt is a chat with a system message and a user message, and no assistant answer. The system message is the EGX analyst rule plus the XML format. The user message has the ticker, the date, the 60-session summary, news from seven days through the date, the horizon, and the five allowed labels.

The label is the percent return `(close at T+H - close at T) / close at T`, with the close on date T as the base:

| Label | Rule |
| --- | --- |
| Strong Bullish | return > +5% |
| Slight Bullish | +1% < return <= +5% |
| Neutral | -1% <= return <= +1% |
| Slight Bearish | -5% <= return < -1% |
| Strong Bearish | return < -5% |

On the full file the mix is about 26.7% Neutral, 25.1% Slight Bullish, 24.3% Slight Bearish, 13.9% Strong Bullish, and 10.0% Strong Bearish. Always guessing Neutral scores about 27%. A uniform guess scores about 20%.

The jsonl stays the full set. At train time, `split_by_recent_dates` holds out the most recent 15% of dates for each ticker. That is 23 validation dates and 127 training dates when a ticker has 150 dates. On this file the split is 2,413 train and 437 validation, with no shared dates, and every validation date after that ticker's last training date.

## Rewards

`rewards.py` scores each completion. Completions may be plain strings or TRL's chat-shaped lists.

| Reward | Score |
| --- | --- |
| Format | 1.0 when the text is exactly `<think>...</think><answer>...</answer>`. Otherwise 0.2 for each of: starts with `<think>`, contains `</think>`, contains `<answer>`, ends with `</answer>`. |
| XML count | 0.5 when each of the four tags appears exactly once, otherwise 0. |
| Accuracy | 2.0 when the text inside `<answer>` is the target label, otherwise 0. |

A run that used the older all-or-nothing format reward should not be resumed after this definition.

## Training design

`train_grpo.py` loads `sft/merged_16bit` in 4-bit and attaches a **fresh** LoRA. It does not keep training the SFT adapter. Unsloth is imported before Transformers, and `PatchFastRL("GRPO", FastLanguageModel)` runs before `GRPOTrainer` is imported. vLLM is off.

Dr. GRPO in TRL 0.24.0 is both `loss_type="dr_grpo"` and `scale_rewards=False`. `beta=0` drops the KL term.

The file `config.py` is the 4 GB laptop setup:

| Setting | Value |
| --- | --- |
| LoRA | `r=16`, `alpha=32`, dropout 0, same seven modules as SFT |
| Sequence length | 2,048 |
| Prompt / completion caps | 1,024 / 512 |
| Generations per prompt | 4 in training, 1 at evaluation |
| Learning rate | `5e-6`, cosine, warmup ratio 0.05, weight decay 0.01 |
| Batch | 1 per device, gradient accumulation 8 |
| Max steps | 600 |
| KL beta | 0 |
| Temperature | 1.0 |
| Optimizer | `adamw_8bit` |
| Precision | bfloat16 |
| Seed | 123 |

Evaluation generates one completion per prompt. `eval_loss` stays near zero in that setting because the advantages are zero, so it is not the selection metric. The trainer keeps the checkpoint with the highest `eval_reward`.

The published model was not the 4 GB, 600-step run. That local setup was too slow on an RTX 3050. The weights in `merged_16bit/` come from six epochs on a Colab A100. Each epoch loaded only the best LoRA weights so far and started a new cosine schedule at `5e-6`. Plain resume was avoided because the cosine had already decayed to about `1.5e-7`. Completion length moved from 512 to 768 to 1,024. The last epoch used 1,024 completion tokens, 32 sequences per microbatch, and gradient accumulation 8, which is 64 prompts per optimizer step and 38 steps per epoch.

Best point: epoch 6, step 28, `eval_reward` 1.018, exact label on 26.1% of the 437 validation prompts. `checkpoints/checkpoint-28` is that step. `final_adapter/` has the same weights, but its config records a Colab base path, so other people should load `merged_16bit/` or the Hugging Face repo. `train_log.jsonl` is the epoch 6 log.

## How to run it

From the repository root, after `sft/merged_16bit` exists:

```powershell
python -m grpo.grpo_dataset_gen
python -m grpo.train_grpo
python -m grpo.train_grpo --resume
python -m grpo.train_grpo --export-gguf
```

`grpo_dataset_gen` is resumable and does not call a language model. `train_grpo` needs Unsloth, TRL 0.24.0, and a CUDA PyTorch build. Do not reinstall Unsloth on Windows if that build is already in the environment. `--resume` continues the checkpoint in `grpo/checkpoints/`. `--export-gguf` writes `q4_k_m` under `grpo/gguf/`.

A finished `train_grpo` run exports `grpo/final_adapter` and `grpo/merged_16bit`. The folder already in this project is the epoch-6 step-28 merge, which is the model on Hugging Face.
