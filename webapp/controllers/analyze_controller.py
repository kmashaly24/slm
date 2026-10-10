"""Run the four trained questions on the loaded case."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException

from webapp.models import catalog, inference, prices
from webapp.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest) -> dict:
    try:
        meta = catalog.profile(body.symbol)
        window = prices.get_window(meta["symbol"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        results = inference.run_all(
            meta["symbol"],
            meta["company"],
            window["analysis_date"],
            window["summary"],
            body.news_block,
            body.horizon,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "symbol": meta["symbol"],
        "company": meta["company"],
        "analysis_date": date.fromisoformat(window["analysis_date"]).isoformat(),
        "horizon": body.horizon,
        "results": results,
    }
