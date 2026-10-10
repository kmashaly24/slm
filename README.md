# EGX analyst

An English model for Egyptian Stock Exchange names, plus a local page that runs it. The model reads a ticker, the last 60 sessions, and a week of news. It answers in prose and names one return class.

Published weights:

[karimmashaly/egx-qwen2.5-1.5b-grpo](https://huggingface.co/karimmashaly/egx-qwen2.5-1.5b-grpo)

Those weights are the merged 16-bit GRPO model. The same files are in `grpo/merged_16bit`. The local app loads that folder.

## The four phases

Training is documented in the phase folders. Each README covers how the data was built, the design choices, and the commands to run that phase.

1. [Continued pretraining](mid_training/README.md). QLoRA on `Qwen/Qwen2.5-1.5B` over finance books, articles, and narrative 10-K sections. Best checkpoint: step 400, validation loss 1.7551. Merged weights: `mid_training/phase1/merged_16bit`.
2. [Supervised fine-tuning](sft/README.md). 3,120 Gemini-written analyst chats on EGX names. Three questions: five-day direction, risks and sentiment, and macro effect. Best checkpoint: step 500, validation loss 0.7725. Merged weights: `sft/merged_16bit`.
3. [Dr. GRPO](grpo/README.md). A five-way return class over horizons of 1, 3, 5, or 10 trading days. Best checkpoint: epoch 6, step 28, `eval_reward` 1.018, exact label on 26.1% of 437 held-out prompts. Always guessing Neutral scores about 27% on this mix. Merged weights: `grpo/merged_16bit`, also published at the link above.
4. The local app in `webapp/`, described below.

This is a research checkpoint, not a trading system.

## Run the web app

The app is for one computer. It is not deployed. Use the Python environment that already has the CUDA build of PyTorch and Unsloth. Do not `pip install unsloth` again on Windows: that install can replace CUDA PyTorch with a CPU build.

From the repository root:

```powershell
pip install -r webapp/requirements.txt
python -m webapp.app
```

`webapp/requirements.txt` is FastAPI and Uvicorn. Price download, news fetch, and generation use packages already required by `sft/requirements.txt` (`torch`, `transformers`, `unsloth`, `pandas`, `requests`).

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### What the page does

1. **Search.** Type a company or ticker. Matches are Egyptian names (Yahoo symbols ending in `.CA`, or the Cairo exchange). Trained companies map back to the short symbols used in training, such as `COMI.CA` for CIB and `JUFO.CA` for Juhayna. Other names are matched to a tradable symbol and a sector from the listed-company directory.
2. **Prices.** The last 60 sessions are plotted. The analysis date is the newest session in that download, not today's calendar date. The exchange is closed Friday and Saturday. Thursday 8 October 2026 was also closed for Armed Forces Day, so a weekend visit can show Wednesday 7 October. The chart summary is the same text used in training: start close, last close, change, high, low, and the last 10 closes.
3. **News.** Up to six headlines are collected in each of corporate, sector, and macro, for the seven days through the analysis date. Trained tickers use the same headline rules as training. Other tickers search the company name plus Egypt, the sector name plus Egypt, and the shared Egypt macro query. The model then drops headlines it does not treat as relevant to that stock. The kept lines stay editable.
4. **Analyze.** Choose a horizon of 1, 3, 5, or 10 trading days. The app sends the edited news and the price summary to `grpo/merged_16bit` and shows:
   - the five-day direction answer
   - the macro and sector answer
   - the return class only: Strong Bullish, Slight Bullish, Neutral, Slight Bearish, or Strong Bearish

The risk-and-sentiment question from supervised training is not asked in the app. The two written answers may use up to 1,024 new tokens each. If that limit cuts a sentence, the visible text stops at the last period, so the card does not end mid-sentence. A number such as `25.00` is not treated as the end of a sentence. The return class is parsed out of the completion, and the thinking text is not shown.

The first analysis loads the model with Unsloth in 16-bit. On a 4 GB GPU that uses most of the card. If the load or a generation runs out of memory, the app loads the same folder in 4-bit and continues. Later analyses reuse the loaded weights.

### Code map

| Piece | Role |
| --- | --- |
| `webapp/app.py` | FastAPI app, static files, and `python -m webapp.app` |
| `webapp/models/catalog.py` | EGX search and company or sector lookup |
| `webapp/models/prices.py` | 60-session window and candle summary |
| `webapp/models/news.py` | Wider one-week news fetch |
| `webapp/models/inference.py` | Unsloth load, news filter, two answers, return-class label |
| `webapp/controllers/` | `GET /api/search`, `POST /api/load`, `POST /api/analyze` |
| `webapp/views/` | Dark page, chart, editable news, answer cards |

## Load the model without the app

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "karimmashaly/egx-qwen2.5-1.5b-grpo"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

A local path works the same way: `grpo/merged_16bit`.

To retrain a phase, follow that phase's README. The order is continued pretraining, then supervised fine-tuning, then GRPO. The web app only needs the last merged folder.
