# Local analyst

A local web page for one Egyptian stock at a time. It is not deployed. You search for a name, see the recent price path and a week of news, edit the news if you want, and then read two written answers plus one return class from the merged GRPO model.

The model is `grpo/merged_16bit`, the same weights as [karimmashaly/egx-qwen2.5-1.5b-grpo](https://huggingface.co/karimmashaly/egx-qwen2.5-1.5b-grpo).

## What you do on the page

1. Type a company or ticker. A dropdown lists Egyptian matches. Trained names resolve to the short symbols used in training, such as `COMI.CA` for CIB and `JUFO.CA` for Juhayna. Other names are matched to a tradable `.CA` symbol.
2. Pick one. The page loads that stock. The first load also loads the model, so it can take several minutes.
3. Read the price panel. It plots the last 60 sessions and shows the last close, the change over that window, the high, the low, and the last ten closes. Rises are green and falls are red.
4. Read the news panel. Headlines are grouped into Corporate, Sector, and Macro. You can edit any line or add one. Analyze uses the text you leave in the box.
5. Choose a horizon of 1, 3, 5, or 10 trading days and press Analyze.

Analyze shows three cards:

- **Five-day direction.** A written forecast for the next five sessions.
- **Macro and sector.** How recent macro and sector news bears on this stock.
- **Return class.** Only the label for the horizon you chose: Strong Bullish, Slight Bullish, Neutral, Slight Bearish, or Strong Bearish. The reasoning around that label is not shown.

## What happens underneath

The analysis date is the newest trading session in the Yahoo download, not today's calendar date. The Egyptian exchange is closed Friday and Saturday, and it was also closed on Thursday 8 October 2026, so a weekend visit can show Wednesday 7 October.

News covers the seven days through that session, up to six headlines in each group. Trained tickers use the same company and sector filters as training. Other tickers search the company name plus Egypt, the sector name plus Egypt, and the shared Egypt macro query. The model then keeps the headlines it treats as relevant to that stock. That filter runs for every stock, trained or not.

The two written answers may each use up to 1,024 new tokens. If that limit cuts a sentence, the visible text stops at the last period. A decimal such as `25.00` is not treated as the end of a sentence. If the model finishes earlier, the whole answer is kept.

The first analysis loads the 16-bit weights with Unsloth. On a 4 GB GPU that uses most of the card. If the load or a generation runs out of memory, the app loads the same folder in 4-bit and continues. Later clicks reuse the loaded weights.

## Run

From the repository root, in the environment that already has CUDA PyTorch and Unsloth:

```powershell
pip install -r webapp/requirements.txt
python -m webapp.app
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Do not reinstall Unsloth on Windows; that install can replace the CUDA PyTorch build.

The page is a FastAPI app. `models/` fetches prices, news, and model answers. `controllers/` serves search, load, and analyze. `views/` is the dark page.
