"""Run speculative decoding entirely on the edge.

This script expects two OpenAI-compatible local inference servers:

* draft server:  http://127.0.0.1:8000/v1
* target server: http://127.0.0.1:8080/v1

The target server performs verification locally. No cloud endpoint,
WebSocket, or network tunnel is used.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence

import httpx
from openai import OpenAI


DEFAULT_DRAFT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TARGET_URL = "http://127.0.0.1:8080/v1"
DEFAULT_STOP_SEQUENCES = ("<|endoftext|>", "Human:", "Question:")


def create_client(base_url: str) -> OpenAI:
    """Create an OpenAI client configured for a local HTTP/1.1 vLLM server."""
    return OpenAI(
        base_url=base_url,
        api_key="none",
        http_client=httpx.Client(http1=True, http2=False),
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
) -> str:
    """Generate text using local draft generation and local target verification."""
    if lookahead_window < 1:
        raise ValueError("lookahead_window must be at least 1.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")

    draft_client = create_client(draft_url)
    target_client = create_client(target_url)
    stop_sequences = list(DEFAULT_STOP_SEQUENCES)
    current_text = f"Question: {prompt_text}\nAnswer:"
    total_generated_tokens = 0
    step = 1

    try:
        draft_model = get_model_id(draft_client)
        target_model = get_model_id(target_client)
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to both local inference servers. "
            f"Check the draft server ({draft_url}) and target server ({target_url})."
        ) from exc

    print(f"Draft model:  {draft_model}")
    print(f"Target model: {target_model}")
    print("\nRunning edge-only speculative decoding...")
    print("-" * 95)
    print(f"{'Step':<6} | {'Draft chunk':<35} | {'Accepted':<10} | {'Correction':<15}")
    print("-" * 95)

    while total_generated_tokens < max_tokens:
        remaining_tokens = max_tokens - total_generated_tokens
        draft_response = draft_client.completions.create(
            model=draft_model,
            prompt=current_text,
            temperature=0.0,
            max_tokens=min(lookahead_window, remaining_tokens),
            logprobs=1,
            stop=stop_sequences,
        )
        draft_choice = draft_response.choices[0]
        draft_logprobs = draft_choice.logprobs
        draft_tokens = draft_logprobs.tokens if draft_logprobs else []
        if not draft_tokens:
            break

        # Generate the same local lookahead from the target model and compare
        # token-by-token. This replaces the cloud verification request.
        target_response = target_client.completions.create(
            model=target_model,
            prompt=current_text,
            temperature=0.0,
            max_tokens=len(draft_tokens),
            logprobs=1,
            stop=stop_sequences,
        )
        target_logprobs = target_response.choices[0].logprobs
        target_tokens = target_logprobs.tokens if target_logprobs else []

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
            print("\nGeneration stalled: the target server returned no usable token.")
            break

        current_text += appended_chunk
        total_generated_tokens += accepted + (1 if correction is not None else 0)
        print(
            f"{step:<6} | {appended_chunk!r:<35} | "
            f"{accepted:<10} | {correction!r:<15}"
        )

        if any(stop in appended_chunk for stop in stop_sequences):
            break
        step += 1

    print("-" * 95)
    print(f"\nGenerated text:\n{current_text}\n")
    return current_text


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
    run_edge_only(
        prompt,
        lookahead_window=args.lookahead_window,
        max_tokens=args.max_tokens,
        draft_url=args.draft_url,
        target_url=args.target_url,
    )
    print(f"Elapsed time: {time.perf_counter() - start:.3f} seconds")


if __name__ == "__main__":
    main()