import time

import altair as alt
import httpx
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

API_URL = "http://127.0.0.1:9000/generate"

# Baseline = the target model alone, no draft model.
# Same target server edge_only.py uses (DEFAULT_TARGET_URL there).
TARGET_URL = "http://127.0.0.1:8080/v1"

# Keep these in sync with edge_only.py so both runs see exactly the same input.
PROMPT_TEMPLATE = "Question: {prompt}\nAnswer:"
STOP_SEQUENCES = ["<|endoftext|>", "Human:", "Question:"]

LOOKAHEAD_WINDOW = 5
MAX_TOKENS = 60

EDGE = "Edge speculative"
BASE = "Target only"
METHOD_COLORS = alt.Scale(domain=[EDGE, BASE], range=["#4C78A8", "#F58518"])
OUTCOME_COLORS = alt.Scale(domain=["Accepted", "Rejected"], range=["#54A24B", "#E45756"])
PART_COLORS = alt.Scale(
    domain=["Draft model", "Target model", "Other"],
    range=["#72B7B2", "#F58518", "#BAB0AC"],
)


# ---------------------------------------------------------------- requests


def run_speculative(prompt):
    payload = {
        "prompt": prompt,
        "lookahead_window": LOOKAHEAD_WINDOW,
        "max_tokens": MAX_TOKENS,
    }
    start = time.perf_counter()
    response = requests.post(API_URL, json=payload, timeout=300)
    response.raise_for_status()
    rtt_ms = (time.perf_counter() - start) * 1000
    return response.json(), rtt_ms


def run_baseline(prompt):
    """Target model alone, greedy, streamed so TTFT and per-token timing can be measured."""
    client = OpenAI(
        base_url=TARGET_URL,
        api_key="none",
        http_client=httpx.Client(http1=True, http2=False, timeout=300),
    )
    try:
        model = client.models.list().data[0].id
        token_times = []  # ms since start, one entry per streamed chunk
        text = ""
        usage_tokens = None

        start = time.perf_counter()
        stream = client.completions.create(
            model=model,
            prompt=PROMPT_TEMPLATE.format(prompt=prompt),
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stop=STOP_SEQUENCES,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                token_times.append((time.perf_counter() - start) * 1000)
                text += chunk.choices[0].text
            if getattr(chunk, "usage", None):
                usage_tokens = chunk.usage.completion_tokens
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        client.close()

    tokens = usage_tokens or len(token_times)
    tpot_ms = None
    if tokens > 1 and token_times:
        tpot_ms = (token_times[-1] - token_times[0]) / (tokens - 1)

    return {
        "text": text,
        "tokens": tokens,
        "elapsed_ms": elapsed_ms,
        "ttft_ms": token_times[0] if token_times else None,
        "tpot_ms": tpot_ms,
        "token_times": token_times,
    }


def handle_click():
    prompt = st.session_state.get("prompt", "").strip()
    for key in ("result", "error", "baseline", "baseline_error"):
        st.session_state[key] = None

    if not prompt:
        st.session_state["error"] = "Type a prompt first."
        return

    try:
        result, rtt_ms = run_speculative(prompt)
    except requests.exceptions.RequestException as e:
        st.session_state["error"] = str(e)
        return
    st.session_state["result"] = result
    st.session_state["rtt_ms"] = rtt_ms

    if st.session_state.get("run_baseline"):
        try:
            st.session_state["baseline"] = run_baseline(prompt)
        except Exception as e:  # openai/httpx raise several error types
            st.session_state["baseline_error"] = f"Target-only run failed: {e}"


# ---------------------------------------------------------------- metrics


def edge_metrics(result):
    """Everything here is computed from the per-step records edge_only.py returns."""
    steps = result["steps"]
    tokens = sum(s["new_tokens"] for s in steps)
    first_t = steps[0]["t_ms"]
    after_first = tokens - steps[0]["new_tokens"]

    timeline = [{"method": EDGE, "ms": 0.0, "tokens": 0}]
    running = 0
    for s in steps:
        running += s["new_tokens"]
        timeline.append({"method": EDGE, "ms": s["t_ms"], "tokens": running})

    return {
        "tokens": tokens,
        "elapsed_ms": result.get("elapsed_ms", steps[-1]["t_ms"]),
        "ttft_ms": first_t,  # time to first *chunk*: step 1 can add several tokens at once
        "tpot_ms": (steps[-1]["t_ms"] - first_t) / after_first if after_first > 0 else None,
        "draft_ms": sum(s["draft_ms"] for s in steps),
        "target_ms": sum(s["target_ms"] for s in steps),
        "timeline": timeline,
    }


# ---------------------------------------------------------------- charts


def method_bar(df, title, fmt=".0f"):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("method:N", title=None, axis=alt.Axis(labelAngle=0, labels=False, ticks=False)),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("method:N", scale=METHOD_COLORS, legend=None),
            tooltip=["method:N", alt.Tooltip("value:Q", title=title, format=fmt)],
        )
        .properties(height=220, title=title)
    )


def donut(df, field, color_field, scale, title):
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=f"{field}:Q",
            color=alt.Color(f"{color_field}:N", scale=scale, title=None),
            tooltip=[f"{color_field}:N", alt.Tooltip(f"{field}:Q", format=".0f")],
        )
        .properties(height=260, title=title)
    )


def stacked_steps(df, field, color_field, scale, y_title, title):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("step:O", title="Step", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{field}:Q", title=y_title),
            color=alt.Color(f"{color_field}:N", scale=scale, title=None),
            tooltip=["step:O", f"{color_field}:N", alt.Tooltip(f"{field}:Q", format=".1f")],
        )
        .properties(height=260, title=title)
    )


def fmt_ms(x):
    return "—" if x is None else f"{x:.0f} ms"


# ---------------------------------------------------------------- page

st.title("Edge speculative decoding")

st.text_area("Prompt", key="prompt")
st.checkbox("Also run the target model alone for comparison", value=True, key="run_baseline")
st.button("Send request", on_click=handle_click)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

elif st.session_state.get("result"):
    result = st.session_state["result"]
    steps = result["steps"]

    st.subheader("Generated text")
    st.write(result["generated_text"].strip())

    if not steps:
        st.warning(f"No decoding steps came back (stop reason: {result.get('stop_reason', 'unknown')}).")
        st.stop()

    edge = edge_metrics(result)
    st.caption(
        f"Round trip from this app: {st.session_state['rtt_ms']:.0f} ms · "
        f"generation inside edge_only.py: {edge['elapsed_ms']:.0f} ms · "
        f"{edge['tokens']} tokens · stop reason: {result.get('stop_reason', '—')}"
    )

    # ---------- Edge vs. target only ----------
    st.subheader("Edge speculative vs. target only")
    baseline = st.session_state.get("baseline")

    if st.session_state.get("baseline_error"):
        st.warning(st.session_state["baseline_error"])
    elif not baseline:
        st.info("Tick the comparison checkbox and send again to compare against the target model alone.")
    else:
        speedup = baseline["elapsed_ms"] / edge["elapsed_ms"]
        c1, c2, c3 = st.columns(3)
        c1.metric(EDGE, fmt_ms(edge["elapsed_ms"]))
        c2.metric(BASE, fmt_ms(baseline["elapsed_ms"]))
        c3.metric(
            "Speedup",
            f"{speedup:.2f}×",
            help="Target-only time ÷ edge time. Below 1× means the edge pipeline was slower.",
        )

        comparisons = [
            ("Total time (ms)", edge["elapsed_ms"], baseline["elapsed_ms"], ".0f"),
            ("TTFT (ms)", edge["ttft_ms"], baseline["ttft_ms"], ".0f"),
            ("TPOT (ms / token)", edge["tpot_ms"], baseline["tpot_ms"], ".1f"),
            (
                "Tokens / second",
                edge["tokens"] / (edge["elapsed_ms"] / 1000),
                baseline["tokens"] / (baseline["elapsed_ms"] / 1000) if baseline["tokens"] else None,
                ".1f",
            ),
        ]
        for col, (title, e_val, b_val, fmt) in zip(st.columns(4), comparisons):
            df = pd.DataFrame({"method": [EDGE, BASE], "value": [e_val, b_val]}).dropna()
            col.altair_chart(method_bar(df, title, fmt), use_container_width=True)

        # Legend for the four small charts above
        st.markdown(
            "<span style='color:#4C78A8'>■</span> Edge speculative &nbsp;&nbsp;"
            "<span style='color:#F58518'>■</span> Target only",
            unsafe_allow_html=True,
        )
        st.caption(
            "Edge TTFT is time to the first *step*, which can add several tokens at once, "
            "so it's an approximation. Target-only TTFT is time to the first streamed token."
        )

        # Tokens generated over time
        base_timeline = [{"method": BASE, "ms": 0.0, "tokens": 0}] + [
            {"method": BASE, "ms": t, "tokens": i + 1} for i, t in enumerate(baseline["token_times"])
        ]
        timeline = pd.DataFrame(edge["timeline"] + base_timeline)
        st.altair_chart(
            alt.Chart(timeline)
            .mark_line(interpolate="step-after", point=True)
            .encode(
                x=alt.X("ms:Q", title="Time since start (ms)"),
                y=alt.Y("tokens:Q", title="Tokens generated"),
                color=alt.Color("method:N", scale=METHOD_COLORS, title=None),
                tooltip=["method:N", alt.Tooltip("ms:Q", format=".0f"), "tokens:Q"],
            )
            .properties(height=300, title="Tokens generated over time"),
            use_container_width=True,
        )

        # With greedy decoding, speculative decoding should reproduce the target's own output.
        if baseline["text"].strip() == result["generated_text"].strip():
            st.success("Same output as the target model alone.")
        else:
            st.warning("Output differs from the target model alone.")
            with st.expander("Compare the two outputs"):
                st.text(f"{EDGE}:\n{result['generated_text'].strip()}")
                st.text(f"{BASE}:\n{baseline['text'].strip()}")

    # ---------- Where the edge time goes ----------
    st.subheader("Where the edge time goes")
    other_ms = max(edge["elapsed_ms"] - edge["draft_ms"] - edge["target_ms"], 0.0)
    parts = pd.DataFrame(
        {
            "part": ["Draft model", "Target model", "Other"],
            "ms": [edge["draft_ms"], edge["target_ms"], other_ms],
        }
    )
    per_step_time = pd.DataFrame(
        [{"step": s["step"], "part": "Draft model", "ms": s["draft_ms"]} for s in steps]
        + [{"step": s["step"], "part": "Target model", "ms": s["target_ms"]} for s in steps]
    )
    left, right = st.columns([1, 2])
    left.altair_chart(donut(parts, "ms", "part", PART_COLORS, "Total"), use_container_width=True)
    right.altair_chart(
        stacked_steps(per_step_time, "ms", "part", PART_COLORS, "ms", "Per step"),
        use_container_width=True,
    )

    # ---------- Draft acceptance ----------
    st.subheader("Draft acceptance")
    per_step_tokens = pd.DataFrame(
        [{"step": s["step"], "outcome": "Accepted", "tokens": s["accepted"]} for s in steps]
        + [
            {"step": s["step"], "outcome": "Rejected", "tokens": len(s["draft_tokens"]) - s["accepted"]}
            for s in steps
        ]
    )
    totals = per_step_tokens.groupby("outcome", as_index=False)["tokens"].sum()
    n_draft = int(totals["tokens"].sum())
    n_accepted = int(totals.loc[totals["outcome"] == "Accepted", "tokens"].sum())
    st.metric(
        "Acceptance rate",
        f"{n_accepted / n_draft:.0%}" if n_draft else "—",
        help=f"{n_accepted} of {n_draft} draft tokens matched the target model.",
    )
    left, right = st.columns([1, 2])
    left.altair_chart(
        donut(totals, "tokens", "outcome", OUTCOME_COLORS, "All draft tokens"),
        use_container_width=True,
    )
    right.altair_chart(
        stacked_steps(per_step_tokens, "tokens", "outcome", OUTCOME_COLORS, "Draft tokens", "Per step"),
        use_container_width=True,
    )

    # ---------- Raw steps ----------
    st.subheader("Steps")
    st.table(
        [
            {
                "step": s["step"],
                "draft tokens": " | ".join(repr(t) for t in s["draft_tokens"]),
                "accepted": s["accepted"],
                "correction": "" if s["correction"] is None else repr(s["correction"]),
                "draft ms": f"{s['draft_ms']:.0f}",
                "target ms": f"{s['target_ms']:.0f}",
            }
            for s in steps
        ]
    )