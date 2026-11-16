"""Streamlit dashboard for the distributed speculative decoding benchmark."""

import importlib.util
import time
from pathlib import Path

import pandas as pd
import streamlit as st


BENCHMARK_PATH = Path(__file__).with_name("true_distributed_benchmark v2.py")
EDGE_ONLY_PATH = Path(__file__).with_name("edge_only.py")


@st.cache_resource
def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("distributed_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def load_edge_only_module():
    spec = importlib.util.spec_from_file_location("edge_only", EDGE_ONLY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load edge-only module from {EDGE_ONLY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st.set_page_config(
    page_title="Distributed Speculative Decoding",
    page_icon="⚡",
    layout="wide",
)

st.title("Distributed Speculative Decoding Benchmark")
st.caption("Compare target-only generation with the local-draft/WebSocket verification pipeline.")

with st.sidebar:
    st.header("Benchmark settings")
    lookahead_window = st.slider(
        "Lookahead window",
        min_value=1,
        max_value=32,
        value=5,
        help="Number of draft tokens proposed before each verification pass.",
    )
    max_tokens = st.slider(
        "Maximum output tokens",
        min_value=1,
        max_value=512,
        value=60,
        step=1,
    )
    include_edge_only = st.checkbox(
        "Run edge-only comparison",
        help="Also run speculative decoding with local draft and target verification.",
    )
    st.info("Expected services: draft :8000, verifier :8001, target :8080.")

prompt = st.text_area(
    "Prompt",
    value="Explain how speculative decoding improves language-model serving.",
    height=150,
)
run_benchmark = st.button("Run benchmark", type="primary", use_container_width=True)

if run_benchmark:
    if not prompt.strip():
        st.error("Enter a prompt before running the benchmark.")
        st.stop()

    try:
        benchmark = load_benchmark_module()
        progress = st.progress(0, text="Connecting to benchmark services...")
        status = st.empty()
        step_rows = []
        edge_step_rows = []

        def update_progress(event):
            if event["type"] == "phase":
                status.info(event["name"])
            elif event["type"] == "step":
                step_rows.append(event)
                progress_value = min(event["total_tokens"] / max_tokens, 1.0)
                progress.progress(progress_value, text=f"Decoding step {event['step']}")
                status.info(
                    f"Accepted {event['accepted']}/{event['draft_count']} draft tokens "
                    f"({event['total_tokens']}/{max_tokens} total)"
                )

        with st.spinner("Running benchmark..."):
            result = benchmark.run_distributed_benchmark(
                prompt.strip(),
                lookahead_window=lookahead_window,
                max_tokens=max_tokens,
                progress_callback=update_progress,
            )
        edge_result = None
        if include_edge_only:
            edge_status = st.empty()
            edge_status.info("Running edge-only comparison...")
            edge_only = load_edge_only_module()
            edge_start = time.perf_counter()
            edge_text = edge_only.run_edge_only(
                prompt.strip(),
                lookahead_window=lookahead_window,
                max_tokens=max_tokens,
                progress_callback=edge_step_rows.append,
            )
            edge_result = {
                "generated_text": edge_text,
                "elapsed_time": time.perf_counter() - edge_start,
                "generated_tokens": (
                    edge_step_rows[-1]["total_tokens"] if edge_step_rows else 0
                ),
                "accepted_tokens": (
                    edge_step_rows[-1]["accepted_total"] if edge_step_rows else 0
                ),
            }
            edge_status.success("Edge-only comparison completed.")
        progress.progress(1.0, text="Benchmark complete")
        status.success("Benchmark completed.")
        st.session_state["benchmark_result"] = result
        st.session_state["benchmark_steps"] = step_rows
        st.session_state["edge_result"] = edge_result
        st.session_state["edge_steps"] = edge_step_rows
    except Exception as exc:
        st.error(f"Benchmark failed: {exc}")

result = st.session_state.get("benchmark_result")
if result:
    st.subheader("Results")
    metrics = st.columns(4)
    metrics[0].metric("Target baseline", f"{result['baseline_time']:.3f}s")
    metrics[1].metric("Distributed", f"{result['speculative_time']:.3f}s")
    metrics[2].metric("Speedup", f"{result['speedup_ratio']:.2f}x")
    metrics[3].metric("Accepted tokens", result["accepted_tokens"])

    edge_result = st.session_state.get("edge_result")
    edge_steps = st.session_state.get("edge_steps", [])
    throughput_rows = [
        {
            "Architecture": "Distributed",
            "Generated tokens/sec": (
                result["accepted_tokens"] / result["speculative_time"]
                if result["speculative_time"] > 0
                else 0.0
            ),
            "Accepted tokens/sec": (
                result["draft_accepted_tokens"] / result["speculative_time"]
                if result["speculative_time"] > 0
                else 0.0
            ),
        }
    ]
    if edge_result:
        throughput_rows.append(
            {
                "Architecture": "Edge-only",
                "Generated tokens/sec": (
                    edge_result["generated_tokens"] / edge_result["elapsed_time"]
                    if edge_result["elapsed_time"] > 0
                    else 0.0
                ),
                "Accepted tokens/sec": (
                    edge_result["accepted_tokens"] / edge_result["elapsed_time"]
                    if edge_result["elapsed_time"] > 0
                    else 0.0
                ),
            }
        )

    st.subheader("Throughput")
    throughput_data = pd.DataFrame(throughput_rows).set_index("Architecture")
    st.bar_chart(throughput_data)

    architecture_rows = [
        {
            "Architecture": "Target baseline",
            "Latency (seconds)": result["baseline_time"],
            "Speedup (x)": 1.0,
        },
        {
            "Architecture": "Distributed",
            "Latency (seconds)": result["speculative_time"],
            "Speedup (x)": result["speedup_ratio"],
        },
    ]
    if edge_result:
        distributed_time = result["speculative_time"]
        edge_time = edge_result["elapsed_time"]
        architecture_rows.append(
            {
                "Architecture": "Edge-only",
                "Latency (seconds)": edge_time,
                "Speedup (x)": (
                    distributed_time / edge_time if edge_time > 0 else 0.0
                ),
            }
        )

    st.subheader("Latency and speedup")
    chart_columns = st.columns(2)
    architecture_data = pd.DataFrame(architecture_rows).set_index("Architecture")
    with chart_columns[0]:
        st.caption("End-to-end latency")
        st.bar_chart(architecture_data["Latency (seconds)"])
    with chart_columns[1]:
        st.caption("Relative speedup")
        st.bar_chart(architecture_data["Speedup (x)"])

    if edge_result:
        st.subheader("Architecture comparison")
        st.dataframe(
            [
                {
                    "Architecture": "Distributed",
                    "Elapsed time": f"{distributed_time:.3f}s",
                    "Relative speed": "1.00x",
                },
                {
                    "Architecture": "Edge-only",
                    "Elapsed time": f"{edge_time:.3f}s",
                    "Relative speed": (
                        f"{distributed_time / edge_time:.2f}x"
                        if edge_time > 0
                        else "N/A"
                    ),
                },
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Edge-only generated text")
        st.text_area(
            "Edge-only output",
            value=edge_result["generated_text"],
            height=220,
            disabled=True,
        )

    st.subheader("Generated text")
    st.text_area("Output", value=result["generated_text"], height=220, disabled=True)

    steps = st.session_state.get("benchmark_steps", [])
    if steps:
        step_data = pd.DataFrame(
            [
                {
                    "Step": step["step"],
                    "Acceptance rate": (
                        step["accepted"] / step["draft_count"]
                        if step["draft_count"]
                        else 0.0
                    ),
                    "Accepted tokens": step["accepted"],
                    "Draft tokens": step["draft_count"],
                    "Draft latency": step["draft_latency"],
                    "Verification latency": step["verification_latency"],
                    "Total step latency": step["step_latency"],
                }
                for step in steps
            ]
        ).set_index("Step")
        st.subheader("Draft-token acceptance")
        st.line_chart(
            step_data[["Acceptance rate"]],
            y_label="Acceptance rate",
        )
        st.subheader("Per-step latency")
        latency_data = step_data[
            ["Draft latency", "Verification latency", "Total step latency"]
        ].rename(
            columns={
                "Draft latency": "Draft generation",
                "Verification latency": "Verification",
                "Total step latency": "Total",
            }
        )
        if edge_steps:
            edge_latency_data = pd.DataFrame(
                {
                    "Edge-only draft": [step["draft_latency"] for step in edge_steps],
                    "Edge-only verification": [
                        step["verification_latency"] for step in edge_steps
                    ],
                    "Edge-only total": [step["step_latency"] for step in edge_steps],
                },
                index=[step["step"] for step in edge_steps],
            )
            latency_data = latency_data.join(edge_latency_data, how="outer")
        st.line_chart(latency_data, y_label="Latency (seconds)")

        st.subheader("Cumulative generation progress")
        progress_data = pd.DataFrame(
            {"Distributed": [step["total_tokens"] for step in steps]},
            index=[step["step"] for step in steps],
        )
        if edge_steps:
            progress_data["Edge-only"] = pd.Series(
                [step["total_tokens"] for step in edge_steps],
                index=[step["step"] for step in edge_steps],
            )
        st.line_chart(progress_data, y_label="Generated tokens")

        st.subheader("Decoding steps")
        st.dataframe(
            [
                {
                    "Step": step["step"],
                    "Accepted": f"{step['accepted']}/{step['draft_count']}",
                    "Total tokens": step["total_tokens"],
                    "Draft latency": f"{step['draft_latency']:.3f}s",
                    "Verification latency": f"{step['verification_latency']:.3f}s",
                    "Total step latency": f"{step['step_latency']:.3f}s",
                    "Appended text": step["text"],
                }
                for step in steps
            ],
            use_container_width=True,
            hide_index=True,
        )
