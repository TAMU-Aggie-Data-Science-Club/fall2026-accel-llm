# Real-Time UI & Data Streaming
Streamlit dashboard for live TTFT / TPOT metrics.

## Independent request playground

This small Streamlit experiment sends JSON to an editable HTTP POST endpoint.
It does not load models or connect to the team's WebSocket verifier. The default
`https://httpbin.org/post` echoes the request; it performs no inference or token
verification.

From the repository root (Python 3.9+):

```sh
cd ui
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Run from `ui/` to load the local dark theme. Open the local URL printed by
Streamlit (usually `http://localhost:8501`). Stop with Ctrl+C.

Edit the URL and JSON, then click **Send request**. Nothing is sent on startup
or when inputs change. Invalid JSON and non-HTTP URLs are rejected before sending.
Requests have a 15-second connection/read inactivity timeout (not a total wall-clock
deadline), and redirects are not followed. HTTP errors retain their response body;
non-JSON bodies are shown as plain text.

**HTTP round-trip time** uses a monotonic performance timer around the POST through
receipt of the response body, excluding JSON display/parsing. It is not TTFT or
model inference latency. For network failures it shows time until failure.
The latest result survives normal Streamlit reruns and input edits within the
session; a browser reload or server restart may reset it.
