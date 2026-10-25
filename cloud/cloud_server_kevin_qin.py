
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()


@app.post("/verify")
def verify(payload: dict):
    if not isinstance(payload.get("text"), str):
        raise HTTPException(status_code=422, detail="text must be a string")
    return {**payload, "text": payload["text"] + " [VERIFIED]"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
