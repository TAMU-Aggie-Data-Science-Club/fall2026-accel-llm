"""Independent HTTP playground; no model or backend imports."""

import json
from time import perf_counter
from urllib.parse import urlsplit

import requests
import streamlit as st


st.set_page_config(page_title="AccelLLM Request Playground", layout="centered")
st.title("AccelLLM Request Playground")
st.caption(
    "httpbin echoes your request for testing. It does not perform model inference "
    "or token verification."
)

target_url = st.text_input("Target URL", value="https://httpbin.org/post")
payload_text = st.text_area(
    "JSON payload",
    value='''{
  "draft_tokens": ["The", "cat", "sat"],
  "text": "The cat sat",
  "logprobs": [-0.1, -0.11, -0.12]
}''',
    height=220,
)

if st.button("Send request", type="primary"):
    try:
        payload = json.loads(payload_text)
        # Reject non-standard NaN/Infinity and numeric overflow before sending.
        encoded_payload = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (ValueError, RecursionError, UnicodeError) as exc:
        st.error(f"Invalid JSON payload: {exc}. No request was sent.")
    else:
        try:
            url = target_url.strip()
            parsed_url = urlsplit(url)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
                raise ValueError("Enter a complete http:// or https:// URL.")
            parsed_url.port  # Validate a supplied port before network access.
        except ValueError as exc:
            st.error(f"Invalid target URL: {exc} No request was sent.")
        else:
            result = {"url": url, "status": None, "body": None, "error": None}
            with st.spinner("Sending request…"):
                started = perf_counter()
                try:
                    response = requests.post(
                        url,
                        data=encoded_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=15,
                        allow_redirects=False,
                    )
                except requests.Timeout:
                    result["error"] = "Request timed out (15-second connection/read timeout)."
                except requests.ConnectionError:
                    result["error"] = "Connection failed. Check the URL, network, and server availability."
                except requests.RequestException as exc:
                    result["error"] = f"Request failed: {exc}"
                finally:
                    result["elapsed_ms"] = (perf_counter() - started) * 1000

                if result["error"] is None:
                    result["status"] = response.status_code
                    result["body"] = response.text
                    if not 200 <= response.status_code < 300:
                        result["error"] = (
                            f"Unsuccessful HTTP response: {response.status_code}. "
                            "Inspect the response below. Redirects are not followed."
                        )
                    try:
                        # Retain serialized JSON so scalar strings/null also render correctly.
                        response.json()
                        result["is_json"] = True
                    except ValueError:
                        result["is_json"] = False
                    response.close()
            st.session_state["latest_result"] = result

if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]
    st.subheader("Latest result")
    st.caption(f"POST {result['url']} — retained until the next request or session reset.")
    st.metric("HTTP round-trip time", f"{result['elapsed_ms']:.1f} ms")
    st.metric("HTTP status code", result["status"] if result["status"] is not None else "No response")
    if result["error"]:
        st.error(result["error"])
    else:
        st.success("Request completed successfully.")
    if result["body"] is not None:
        if result["is_json"]:
            st.json(result["body"], expanded=False)
        else:
            st.warning("The response is not JSON. Raw response body:")
            st.code(result["body"] or "(empty response)", language=None)
    else:
        st.caption("Timing shows elapsed time until failure; no HTTP response was received.")
