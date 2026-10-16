"""Minimal RPC-style server that tags submitted text as verified."""

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.post("/verify")
async def verify(payload: dict[str, Any]) -> dict[str, Any]:
    """Append the verification tag to the payload's text and return it."""
    text = payload.get("text")
    if not isinstance(text, str):
        raise HTTPException(status_code=422, detail="Payload must contain a string 'text' field")

    return {**payload, "text": f"{text} [VERIFIED]"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
