
import time
import streamlit as st
import requests


VERIFY_URL = "https://httpbin.org/post"
#VERIFY_URL = "http://127.0.0.1:8000/verify"

PAYLOAD = {
    # placeholder values; request shape (edge -> cloud)
    "draft_tokens": ["The", "cat", "sat"],
    "text": "The cat sat",
    "logprobs": [-0.10, -0.11, -0.12],
}

def handle_click():
    start = time.perf_counter()
    try:
        response = requests.post(VERIFY_URL, json=PAYLOAD, timeout=10)
        response.raise_for_status()
        result = response.json()

        #roundtrip time calculation + save the result
        st.session_state["rtt_ms"] = (time.perf_counter() - start)*1000
        st.session_state["result"] = result
        st.session_state["error"] = None

    except requests.exceptions.RequestException as e:
        st.session_state["rtt_ms"] = (time.perf_counter() - start) * 1000
        st.session_state["result"] = None
        st.session_state["error"] = str(e)




st.title("Streamlit button to HTTP request")
st.caption(f"Target: {VERIFY_URL}")
st.button("Send request", type="primary", on_click=handle_click)

#return what's stored in session_state
if st.session_state.get("rtt_ms") is not None:
    st.metric("Round-trip time", f"{st.session_state['rtt_ms']:.1f} ms")

if st.session_state.get("error"):
    st.error(st.session_state["error"])

elif st.session_state.get("result"):
    st.json(st.session_state["result"])