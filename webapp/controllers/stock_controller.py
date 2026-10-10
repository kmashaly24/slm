"""Search and case loading."""

from __future__ import annotations

from datetime import date

import requests
from fastapi import APIRouter, HTTPException

from webapp.models import catalog, inference, news, prices
from webapp.schemas import LoadRequest, LoadResponse, SearchHit

router = APIRouter()


@router.get("/api/search", response_model=list[SearchHit])
def search_stocks(q: str = "") -> list[dict]:
    try:
        return catalog.search(q)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Quote search is unavailable") from exc


@router.post("/api/load", response_model=LoadResponse)
def load_stock(body: LoadRequest) -> dict:
    try:
        meta = catalog.profile(body.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Company lookup is unavailable") from exc
    try:
        window = prices.get_window(meta["symbol"])
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    analysis_date = date.fromisoformat(window["analysis_date"])
    candidates = news.get_block(meta["symbol"], analysis_date, meta["company"], meta["sector"])
    try:
        news_block = inference.select_news(
            meta["symbol"],
            meta["company"],
            meta["sector"],
            window["analysis_date"],
            window["summary"],
            candidates,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "symbol": meta["symbol"],
        "company": meta["company"],
        "sector": meta["sector"],
        "trained": meta["trained"],
        "analysis_date": window["analysis_date"],
        "chart": {"dates": window["dates"], "closes": window["closes"]},
        "summary": window["summary"],
        "news_block": news_block,
    }
