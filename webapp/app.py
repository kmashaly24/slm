"""FastAPI app for the local EGX analyst."""

from __future__ import annotations

import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.config import HOST, PORT, VIEWS_DIR
from webapp.controllers.analyze_controller import router as analyze_router
from webapp.controllers.stock_controller import router as stock_router

app = FastAPI(title="EGX Local Analyst")
app.mount("/static", StaticFiles(directory=VIEWS_DIR / "static"), name="static")
app.include_router(stock_router)
app.include_router(analyze_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(VIEWS_DIR / "templates" / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("webapp.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
