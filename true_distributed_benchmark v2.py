import time
import json
import httpx
from websockets.sync.client import connect
from openai import OpenAI

def _http_side(url):
    if url.port == 8000:
        return "LOCAL DRAFT"
    if url.port == 8080:
        return "HPRC TARGET"
    return "UNKNOWN"


def _log_request(request):
    print(f"[HTTP -> {_http_side(request.url)}] {request.method} {request.url.path}")


def _log_response(response):
    response.read()
    print(f"[HTTP <- {_http_side(response.url)}] {response.status_code} {response.url.path}")


# Force HTTP/1.1 to match vLLM's socket architecture over the SSH tunnel.
# Hooks show a compact request/response trace for both model endpoints.
custom_http_client = httpx.Client(
    http1=True,
    http2=False,
    event_hooks={
        "request": [_log_request],
        "response": [_log_response],
    },
)

# Local Draft Engine (Edge)
local_draft_client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none",
    http_client=custom_http_client
)

# Remote Target Engine (HPRC - tunneled for baseline testing)
hprc_target_client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="none",
    http_client=custom_http_client
)

# Remote FastAPI WebSocket Verifier (HPRC)
FASTAPI_WS_URL = "ws://127.0.0.1:8001/ws/verify"

def run_distributed_benchmark(prompt_text, lookahead_window=5, max_tokens=60):
    try:
        local_model_str = local_draft_client.models.list().data[0].id
        hprc_model_str = hprc_target_client.models.list().data[0].id
    except Exception as e:
        print(f"❌ Connection failed. Ensure ports 8000, 8001, and 8080 are active. Error: {e}")
        return

    # =========================================================================
    # PHASE 1: TARGET MODEL ALONE (NETWORK BASELINE)
    # =========================================================================
    print("⏳ Running Phase 1: Target Model Alone (Network Baseline Metric)...")
    baseline_start = time.time()
    
    baseline_resp = hprc_target_client.completions.create(
        model=hprc_model_str,
        prompt=f"Question: {prompt_text}\nAnswer:",
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["<|endoftext|>", "Human:", "Question:"]
    )
    
    baseline_time = time.time() - baseline_start
    print(f"✅ Baseline completed in {baseline_time:.3f} seconds.")

    # =========================================================================
    # PHASE 2: DISTRIBUTED SPECULATIVE DECODING (SINGLE-PASS PARALLEL VERIFY)
    # =========================================================================
    print("\n⏳ Running Phase 2: Distributed WebSocket Speculative Decoding Loop...")
    spec_start = time.time()
    
    current_text = f"Question: {prompt_text}\nAnswer:"
    stop_sequences = ["<|endoftext|>", "Human:", "Question:"]
    step = 1
    total_accepted_tokens = 0

    print("\n" + "=" * 72)
    print("STEP | DRAFT TOKENS")
    print("=" * 72)

    try:
        with connect(FASTAPI_WS_URL) as ws:
            while total_accepted_tokens < max_tokens:
                # 1. Generate Local Draft + Extract Logprobs q(x)
                draft_resp = local_draft_client.completions.create(
                    model=local_model_str,
                    prompt=current_text,
                    temperature=0.0,
                    max_tokens=lookahead_window,
                    logprobs=1,
                    stop=stop_sequences
                )
                draft_tokens = draft_resp.choices[0].logprobs.tokens
                draft_logprobs = draft_resp.choices[0].logprobs.token_logprobs
                
                if not draft_tokens:
                    break

                # 2. Send tokens AND logprobs over open WebSocket
                verify_payload = {
                    "prompt": current_text,
                    "draft_tokens": draft_tokens,
                    "draft_logprobs": draft_logprobs
                }
                
                ws.send(json.dumps(verify_payload))
                cloud_resp = json.loads(ws.recv())
                
                accepted_in_this_block = cloud_resp["accepted_count"]
                correction_token = cloud_resp["correction_token"]

                # 3. Process the Cloud's Verdict Locally
                appended_chunk = "".join(draft_tokens[:accepted_in_this_block])

                token_details = "\n".join(
                    f"      {index:>2}. {token!r:<18} logprob={logprob:.4f}"
                    for index, (token, logprob) in enumerate(
                        zip(draft_tokens, draft_logprobs), start=1
                    )
                )

                if correction_token:
                    appended_chunk += correction_token
                    total_accepted_tokens += 1
                    if accepted_in_this_block == len(draft_tokens):
                        result = f"full match + bonus {correction_token!r}"
                    else:
                        result = f"corrected with {correction_token!r}"
                else:
                    result = "full block match"

                current_text += appended_chunk
                total_accepted_tokens += accepted_in_this_block

                print(f"{step:>4} |")
                print(token_details)
                print(
                    f"      accepted={accepted_in_this_block}/{len(draft_tokens)}"
                    f" | {result}"
                )
                
                # 4. Stop & Anti-Stall Checks
                if any(s in appended_chunk for s in stop_sequences):
                    print("\n🛑 Stop sequence encountered. Ending generation.")
                    break
                    
                if accepted_in_this_block == 0 and not correction_token:
                    print("\n🛑 Pipeline stalled: Target model returned no new tokens. Ending generation.")
                    break
                    
                step += 1

    except Exception as e:
        print(f"❌ WebSocket Connection Failed: {e}")

    spec_time = time.time() - spec_start

    # =========================================================================
    # PHASE 3: CALCULATE SPEEDUP METRICS
    # =========================================================================
    speedup_ratio = baseline_time / spec_time if spec_time > 0 else 0

    print("-" * 95)
    print(f"\n📊 FINAL DISTRIBUTED BENCHMARK METRICS:")
    print(f"  • Network Baseline Time (Target Alone): {baseline_time:.3f} seconds")
    print(f"  • Distributed Architecture Time:        {spec_time:.3f} seconds")
    print(f"  • Net Wall-clock Acceleration Factor:   {speedup_ratio:.2f}x speedup ratio")
    print("-" * 95)
    print(f"\n📝 FINAL GENERATED TEXT:\n{current_text}\n")

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    run_distributed_benchmark(prompt, lookahead_window=5, max_tokens=60)