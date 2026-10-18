from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/verify")
async def verift(payload: dict[str, Any]):
    text = str(payload.get("text", ""))
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Payload must include a non-empty string field 'text'.")
    payload["text"] = f"{text} [VERIFIED]".strip()
    return payload

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)