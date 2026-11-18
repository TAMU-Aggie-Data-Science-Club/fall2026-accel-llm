"""FastAPI wrapper around edge_only.run_edge_only().

Exposes edge_only.py as an HTTP endpoint so the Streamlit app (or anything
else) can call it with a plain POST request.

Run:
    uvicorn api:app --port 9000

Endpoints:
    GET  /health    -> quick check that the API is up
    POST /generate  -> body {"prompt": "...", "lookahead_window": 5, "max_tokens": 60}
"""

from __future__ import annotations

import os

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from edge_only import DEFAULT_DRAFT_URL, DEFAULT_TARGET_URL, run_edge_only


DRAFT_URL = os.getenv("DRAFT_URL", DEFAULT_DRAFT_URL)
TARGET_URL = os.getenv("TARGET_URL", DEFAULT_TARGET_URL)

app = FastAPI(title="Edge speculative decoding API")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    lookahead_window: int = Field(5, ge=1, le=32)
    max_tokens: int = Field(60, ge=1, le=1024)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "draft_url": DRAFT_URL, "target_url": TARGET_URL}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    try:
        return run_edge_only(
            req.prompt,
            lookahead_window=req.lookahead_window,
            max_tokens=req.max_tokens,
            draft_url=DRAFT_URL,
            target_url=TARGET_URL,
            verbose=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Raised by edge_only when it can't reach the model servers.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except openai.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Model server error: {exc}") from exc
    