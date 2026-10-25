"""Week 2 onboarding evidence on a local (edge) machine.

Runs four checks with Hugging Face Transformers (vLLM has no native macOS build):
  run      one prompt -> response, wall-clock latency, local memory
  context  memory as the prompt (KV cache) grows
  logprobs top-k logprobs for a single generated token
  spec     target model alone vs. target + draft (assisted / speculative decoding)

Usage:
  .venv/bin/python edge/local_onboarding.py all
  .venv/bin/python edge/local_onboarding.py spec --max-new-tokens 200
"""
import argparse
import os
import platform
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = "models/Qwen1.5-1.8B-Chat"
DRAFT = "models/Qwen1.5-0.5B-Chat"
PROMPT = "Explain in three sentences why speculative decoding makes LLM inference faster."

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
GB = 1024 ** 3


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def memory_report():
    rss = psutil.Process(os.getpid()).memory_info().rss / GB
    if DEVICE == "mps":
        return (f"MPS allocated {torch.mps.current_allocated_memory() / GB:.2f} GB | "
                f"MPS driver {torch.mps.driver_allocated_memory() / GB:.2f} GB | process RSS {rss:.2f} GB")
    if DEVICE == "cuda":
        return f"VRAM allocated {torch.cuda.memory_allocated() / GB:.2f} GB | process RSS {rss:.2f} GB"
    return f"process RSS {rss:.2f} GB"


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()
    return tok, model


def chat_ids(tok, text):
    msgs = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return {k: v.to(DEVICE) for k, v in ids.items()}


def timed_generate(model, inputs, max_new_tokens, **kw):
    sync()
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kw)
    sync()
    return out, time.perf_counter() - start


def header(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def run(tok, model, args):
    header(f"LOCAL RUN | {TARGET} | {DEVICE} {DTYPE} | {platform.machine()} "
           f"{psutil.virtual_memory().total / GB:.0f} GB RAM")
    inputs = chat_ids(tok, PROMPT)
    timed_generate(model, inputs, 8)  # warm-up (Metal kernel compile)
    out, secs = timed_generate(model, inputs, args.max_new_tokens)
    new = out[0, inputs["input_ids"].shape[1]:]
    print(f"PROMPT: {PROMPT}")
    print(f"RESPONSE: {tok.decode(new, skip_special_tokens=True).strip()}")
    print(f"Wall-clock latency: {secs:.2f} s for {len(new)} tokens ({len(new) / secs:.1f} tok/s)")
    print(f"Memory: {memory_report()}")


def context(tok, model, args):
    header("CONTEXT LENGTH vs MEMORY (1 new token per prompt)")
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    kv_bytes_per_token = 2 * cfg.num_hidden_layers * kv_heads * head_dim * torch.finfo(DTYPE).bits // 8
    print(f"KV cache per token = 2 (K,V) x {cfg.num_hidden_layers} layers x {kv_heads} heads x "
          f"{head_dim} dim x {torch.finfo(DTYPE).bits // 8} B = {kv_bytes_per_token / 1024:.0f} KB")
    print(f"{'ctx tokens':>10} | {'theoretical KV':>14} | {'MPS/VRAM in use':>15} | {'prefill s':>9}")
    filler = "The quick brown fox jumps over the lazy dog. "
    for n in args.contexts:
        ids = tok(filler * (n // 10 + 1), return_tensors="pt").input_ids[:, :n].to(DEVICE)
        if DEVICE == "mps":
            torch.mps.empty_cache()
        out, secs = timed_generate(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, 1)
        used = (torch.mps.driver_allocated_memory() if DEVICE == "mps"
                else torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0) / GB
        print(f"{n:>10} | {n * kv_bytes_per_token / GB:>11.2f} GB | {used:>12.2f} GB | {secs:>9.2f}")
        del out


def logprobs(tok, model, args):
    header(f"TOP-{args.top_k} LOGPROBS FOR ONE GENERATED TOKEN")
    prompt = "The capital of Texas is"
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    top = torch.topk(lp, args.top_k)
    print(f"prompt: {prompt!r}")
    print(f"generated token: {tok.decode(top.indices[0])!r}")
    for rank, (v, i) in enumerate(zip(top.values, top.indices), 1):
        print(f"  {rank}. token_id={i.item():>6}  token={tok.decode(i)!r:<12} logprob={v.item():8.4f}  "
              f"p={v.exp().item():.4f}")


def spec(tok, model, args):
    header(f"SPECULATIVE DECODING | target {TARGET} | draft {DRAFT} | greedy, {args.max_new_tokens} tokens")
    _, draft = load(DRAFT)
    inputs = chat_ids(tok, PROMPT)
    timed_generate(model, inputs, 8)
    timed_generate(model, inputs, 8, assistant_model=draft)
    base_out, base_s = timed_generate(model, inputs, args.max_new_tokens)
    spec_out, spec_s = timed_generate(model, inputs, args.max_new_tokens, assistant_model=draft,
                                      num_assistant_tokens=args.lookahead,
                                      num_assistant_tokens_schedule="constant")
    n = base_out.shape[1] - inputs["input_ids"].shape[1]
    m = spec_out.shape[1] - inputs["input_ids"].shape[1]
    print(f"Target alone:                 {base_s:6.2f} s  ({n} tokens, {n / base_s:.1f} tok/s)")
    print(f"Target + draft (k={args.lookahead}):       {spec_s:6.2f} s  ({m} tokens, {m / spec_s:.1f} tok/s)")
    print(f"Speedup: {base_s / spec_s:.2f}x | identical output (lossless): {torch.equal(base_out, spec_out)}")
    print(f"Memory with both models loaded: {memory_report()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section", choices=["run", "context", "logprobs", "spec", "all"])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--contexts", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--lookahead", type=int, default=5)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--draft", default=DRAFT)
    args = p.parse_args()
    globals().update(TARGET=args.target, DRAFT=args.draft)

    tok, model = load(TARGET)
    sections = ["run", "logprobs", "spec", "context"] if args.section == "all" else [args.section]
    for s in sections:
        globals()[s](tok, model, args)


if __name__ == "__main__":
    main()
