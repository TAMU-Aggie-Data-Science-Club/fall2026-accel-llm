"""Run speculative decoding entirely on the edge.

This script expects two OpenAI-compatible local inference servers:

* draft server:  http://127.0.0.1:8000/v1
* target server: http://127.0.0.1:8080/v1

The target server performs verification locally. No cloud endpoint,
WebSocket, or network tunnel is used.

`run_edge_only()` returns a dict with the generated text plus one record per43
decoding step, so it can be called from the CLI, from api.py (FastAPI), or
from anywhere else that wants structured results.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx
from openai import OpenAI


DEFAULT_DRAFT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TARGET_URL = "http://127.0.0.1:8080/v1"
DEFAULT_STOP_SEQUENCES = ("<|endoftext|>", "Human:", "Question:")
PROMPT_TEMPLATE = "Question: {prompt}\nAnswer:"


def create_client(base_url: str, timeout: float = 120.0) -> OpenAI:
    """Create an OpenAI client configured for a local HTTP/1.1 vLLM server."""
    return OpenAI(
        base_url=base_url,
        api_key="none",
        http_client=httpx.Client(http1=True, http2=False, timeout=timeout),
    )


def get_model_id(client: OpenAI) -> str:
    """Return the first model exposed by a local inference server."""
    models = client.models.list().data
    if not models:
        raise RuntimeError("The inference server returned no available models.")
    return models[0].id


def run_edge_only(
    prompt_text: str,
    lookahead_window: int = 5,
    max_tokens: int = 60,
    draft_url: str = DEFAULT_DRAFT_URL,
    target_url: str = DEFAULT_TARGET_URL,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate text using local draft generation and local target verification.

    Returns a dict:
        prompt, full_text, generated_text, draft_model, target_model,
        steps (list of per-step dicts), total_generated_tokens,
        total_draft_tokens, total_accepted_tokens, acceptance_rate,
        ttft_ms, tpot_ms, elapsed_ms, stop_reason

    Timing notes:
        ttft_ms  = time until the first step's tokens were appended. A step
                   can add several tokens at once, so this is "time to first
                   chunk", an approximation of TTFT.
        tpot_ms  = average time per token after the first step.
    """
    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt_text must not be empty.")
    if lookahead_window < 1:
        raise ValueError("lookahead_window must be at least 1.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")

    draft_client = create_client(draft_url)
    target_client = create_client(target_url)
    try:
        return _generate(
            draft_client,
            target_client,
            prompt_text,
            lookahead_window,
            max_tokens,
            draft_url,
            target_url,
            verbose,
        )
    finally:
        draft_client.close()
        target_client.close()


def _generate(
    draft_client: OpenAI,
    target_client: OpenAI,
    prompt_text: str,
    lookahead_window: int,
    max_tokens: int,
    draft_url: str,
    target_url: str,
    verbose: bool,
) -> Dict[str, Any]:
    stop_sequences = list(DEFAULT_STOP_SEQUENCES)
    initial_text = PROMPT_TEMPLATE.format(prompt=prompt_text)
    current_text = initial_text
    total_generated_tokens = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    steps: List[Dict[str, Any]] = []
    stop_reason = "max_tokens"
    step = 1

    try:
        draft_model = get_model_id(draft_client)
        target_model = get_model_id(target_client)
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to both local inference servers. "
            f"Check the draft server ({draft_url}) and target server ({target_url})."
        ) from exc

    if verbose:
        print(f"Draft model:  {draft_model}")
        print(f"Target model: {target_model}")
        print("\nRunning edge-only speculative decoding...")
        print("-" * 95)
        print(f"{'Step':<6} | {'Draft chunk':<35} | {'Accepted':<10} | {'Correction':<15}")
        print("-" * 95)

    start = time.perf_counter()

    while total_generated_tokens < max_tokens:
        remaining_tokens = max_tokens - total_generated_tokens

        t0 = time.perf_counter()
        draft_response = draft_client.completions.create(
            model=draft_model,
            prompt=current_text,
            temperature=0.0,
            max_tokens=min(lookahead_window, remaining_tokens),
            logprobs=1,
            stop=stop_sequences,
        )
        draft_ms = (time.perf_counter() - t0) * 1000

        draft_choice = draft_response.choices[0]
        draft_logprobs = draft_choice.logprobs
        draft_tokens = list(draft_logprobs.tokens) if draft_logprobs and draft_logprobs.tokens else []
        if not draft_tokens:
            stop_reason = "draft_returned_no_tokens"
            break

        # Generate the same local lookahead from the target model and compare
        # token-by-token. This replaces the cloud verification request.
        t0 = time.perf_counter()
        target_response = target_client.completions.create(
            model=target_model,
            prompt=current_text,
            temperature=0.0,
            max_tokens=len(draft_tokens),
            logprobs=1,
            stop=stop_sequences,
        )
        target_ms = (time.perf_counter() - t0) * 1000

        target_logprobs = target_response.choices[0].logprobs
        target_tokens = list(target_logprobs.tokens) if target_logprobs and target_logprobs.tokens else []

        accepted = 0
        correction: Optional[str] = None
        for draft_token, target_token in zip(draft_tokens, target_tokens):
            if draft_token != target_token:
                correction = target_token
                break
            accepted += 1

        if accepted == len(draft_tokens) and len(target_tokens) > accepted:
            correction = target_tokens[accepted]

        appended_chunk = "".join(draft_tokens[:accepted])
        if correction is not None:
            appended_chunk += correction

        if not appended_chunk:
            stop_reason = "stalled"
            if verbose:
                print("\nGeneration stalled: the target server returned no usable token.")
            break

        new_tokens = accepted + (1 if correction is not None else 0)
        current_text += appended_chunk
        total_generated_tokens += new_tokens
        total_draft_tokens += len(draft_tokens)
        total_accepted_tokens += accepted

        steps.append(
            {
                "step": step,
                "draft_tokens": draft_tokens,
                "target_tokens": target_tokens,
                "accepted": accepted,
                "accepted_tokens": draft_tokens[:accepted],
                "correction": correction,
                "appended": appended_chunk,
                "new_tokens": new_tokens,
                "draft_ms": round(draft_ms, 2),
                "target_ms": round(target_ms, 2),
                "t_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )

        if verbose:
            print(
                f"{step:<6} | {appended_chunk!r:<35} | "
                f"{accepted:<10} | {correction!r:<15}"
            )

        if any(stop in appended_chunk for stop in stop_sequences):
            stop_reason = "stop_sequence"
            break
        step += 1

    elapsed_ms = (time.perf_counter() - start) * 1000

    ttft_ms: Optional[float] = steps[0]["t_ms"] if steps else None
    tpot_ms: Optional[float] = None
    if steps:
        tokens_after_first = total_generated_tokens - steps[0]["new_tokens"]
        if tokens_after_first > 0:
            tpot_ms = round((steps[-1]["t_ms"] - steps[0]["t_ms"]) / tokens_after_first, 2)

    if verbose:
        print("-" * 95)
        print(f"\nGenerated text:\n{current_text}\n")

    return {
        "prompt": prompt_text,
        "full_text": current_text,
        "generated_text": current_text[len(initial_text):],
        "draft_model": draft_model,
        "target_model": target_model,
        "steps": steps,
        "total_generated_tokens": total_generated_tokens,
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted_tokens,
        "acceptance_rate": (
            round(total_accepted_tokens / total_draft_tokens, 4) if total_draft_tokens else None
        ),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "elapsed_ms": round(elapsed_ms, 2),
        "stop_reason": stop_reason,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speculative decoding with draft and target models on the edge."
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to complete.")
    parser.add_argument("--draft-url", default=DEFAULT_DRAFT_URL)
    parser.add_argument("--target-url", default=DEFAULT_TARGET_URL)
    parser.add_argument("--lookahead-window", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    prompt = args.prompt or input("Enter prompt: ")
    start = time.perf_counter()
    result = run_edge_only(
        prompt,
        lookahead_window=args.lookahead_window,
        max_tokens=args.max_tokens,
        draft_url=args.draft_url,
        target_url=args.target_url,
    )
    print(f"Acceptance rate: {result['acceptance_rate']}")
    print(f"Elapsed time: {time.perf_counter() - start:.3f} seconds")


if __name__ == "__main__":
    main()