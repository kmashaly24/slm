"""Request and response bodies for the local API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    symbol: str
    name: str


class LoadRequest(BaseModel):
    symbol: str = Field(min_length=1)


class ChartSeries(BaseModel):
    dates: list[str]
    closes: list[float]


class LoadResponse(BaseModel):
    symbol: str
    company: str
    sector: str
    trained: bool
    analysis_date: str
    chart: ChartSeries
    summary: str
    news_block: str


class AnalyzeRequest(BaseModel):
    symbol: str = Field(min_length=1)
    news_block: str = ""
    horizon: Literal[1, 3, 5, 10] = 5


class ResultCard(BaseModel):
    key: str
    question: str
    response: str


class AnalyzeResponse(BaseModel):
    symbol: str
    company: str
    analysis_date: str
    horizon: int
    results: list[ResultCard]
