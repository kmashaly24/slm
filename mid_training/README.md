# Continued pretraining

This phase continues pretraining `Qwen/Qwen2.5-1.5B` on finance text before any Egyptian stock questions. The method is QLoRA: the base model stays in 4-bit, and a LoRA adapter is the only part that is trained. The best adapter is then merged into 16-bit weights for supervised fine-tuning.

Code is in `phase1/`. Raw text is in `mid_training_data/`.

Download [mid_training_data.zip](https://drive.google.com/file/d/1tfqzLuzZm_eZN9WDQHbovIX0sDZxg61z/view?usp=sharing) and unzip it in this directory.
## How the data was created

`phase1/build_dataset.py` reads three local sources and packs them into 2,048-token sequences. The target size is 30 million tokens. Books and articles are counted first. Filings fill whatever budget remains.

| Source | Folder | What is kept |
| --- | --- | --- |
| Books | `mid_training_data/books/` | Page text from `pages.json`. Blocks shorter than 20 words are dropped. A block is treated as a table, and dropped, when at least 30% of its symbols are digits. |
| Articles | `mid_training_data/articles/` | A 10% sample. An article must have at least 300 words. |
| Filings | `mid_training_data/corpus/` | Narrative 10-K sections only: business (`section_1`), risk factors (`section_1A`), properties (`section_2`), legal proceedings (`section_3`), and MD&A (`section_7`, `section_7A`). A section needs at least 40 words. |

`phase1/packing.py` tokenizes those documents with the Qwen tokenizer and packs them so each training row is one 2,048-token sequence. The packed set is saved to `phase1/packed_dataset/`. `phase1/token_report.json` records how many tokens came from each source.

There are no questions, labels, or EGX tickers in this phase.

## Training design

Settings live in `phase1/config.py`.

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B` |
| Method | Unsloth QLoRA, 4-bit load, bfloat16 compute |
| LoRA | `r=32`, `alpha=64`, dropout 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Sequence length | 2,048 |
| Epochs | 1 |
| Learning rate | `2e-4`, cosine, warmup ratio 0.03, weight decay 0.01 |
| Batch | 1 per device, gradient accumulation 32 |
| Optimizer | `adamw_8bit` |
| Precision | bfloat16. fp16 is off because Unsloth's fast LoRA kernel crashes when the dtypes mix. |
| Validation | 5% of packed rows |
| Log / eval / save | every 20 / 40 / 40 steps |
| Seed | 42 |

`phase1/train.py` imports Unsloth before Transformers. It also sets `USE_TF=0` so a broken TensorFlow install cannot be imported. Loss is plain causal-language-model loss on the packed tokens. A callback prints perplexity from the loss.

The best validation checkpoint from the run that fed the later phases is `phase1/checkpoints/checkpoint-400`, with validation loss 1.7551. `phase1/final_adapter/` is that adapter. `phase1/merged_16bit/` is the adapter merged into 16-bit weights. Supervised fine-tuning starts from that merged model.

## How to run it

Use a CUDA build of PyTorch. Do not `pip install unsloth` on this Windows machine if Unsloth is already present: that install can replace the CUDA PyTorch build with a CPU build.

From `mid_training/`, so Python can import the `phase1` package:

```powershell
pip install -r phase1/requirements.txt
python -m phase1.build_dataset
python -m phase1.train
python -m phase1.train --resume
```

`--resume` continues from the latest checkpoint in `phase1/checkpoints/`. The first run downloads `Qwen/Qwen2.5-1.5B`.

After training, merge the adapter before SFT. That step is `python -m sft.merge_phase1` from the repository root, described in [sft/README.md](../sft/README.md).
