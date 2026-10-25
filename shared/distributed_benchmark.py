import time
import json
import httpx
from websockets.sync.client import connect
from openai import OpenAI

# Force HTTP/1.1 to match vLLM's socket architecture over the SSH tunnel
custom_http_client = httpx.Client(http1=True, http2=False)

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

# Remote FastAPI WebSocket Verifier (HPRC - tunneled for distributed pipeline)
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
    # PHASE 2: DISTRIBUTED SPECULATIVE DECODING (EDGE + CLOUD WEBSOCKET)
    # =========================================================================
    print("\n⏳ Running Phase 2: Distributed WebSocket Speculative Decoding Loop...")
    spec_start = time.time()
    
    current_text = f"Question: {prompt_text}\nAnswer:"
    stop_sequences = ["<|endoftext|>", "Human:", "Question:"]
    step = 1
    total_accepted_tokens = 0

    print("-" * 95)
    print(f"{'Step':<6} | {'Draft Chunk Verified':<30} | {'Accepted':<10} | {'Correction':<12} | {'Action'}")
    print("-" * 95)

    try:
        # Open a single persistent connection for the entire generation
        with connect(FASTAPI_WS_URL) as ws:
            while total_accepted_tokens < max_tokens:
                # 1. Generate Local Draft (Edge)
                draft_resp = local_draft_client.completions.create(
                    model=local_model_str,
                    prompt=current_text,
                    temperature=0.0,
                    max_tokens=lookahead_window,
                    logprobs=1,
                    stop=stop_sequences
                )
                draft_tokens = draft_resp.choices[0].logprobs.tokens
                
                if not draft_tokens:
                    break

                # 2. Verify via open WebSocket
                verify_payload = {
                    "prompt": current_text,
                    "draft_tokens": draft_tokens
                }
                
                ws.send(json.dumps(verify_payload))
                cloud_resp = json.loads(ws.recv())
                
                accepted_in_this_block = cloud_resp["accepted_count"]
                correction_token = cloud_resp["correction_token"]

                # 3. Process the Cloud's Verdict Locally
                raw_draft_proposal = "".join(draft_tokens)
                appended_chunk = "".join(draft_tokens[:accepted_in_this_block])
                    
                if correction_token:
                    appended_chunk += correction_token
                    total_accepted_tokens += 1
                    action = "Target Corrected Draft"
                    display_correction = repr(correction_token)
                else:
                    action = "Full Block Match"
                    display_correction = "None"

                current_text += appended_chunk
                total_accepted_tokens += accepted_in_this_block

                print(f"{step:<6} | {repr(raw_draft_proposal):<35} | {accepted_in_this_block:<10} | {display_correction:<12} | {action}")
                
                # 4. BULLETPROOF STOP CHECKS (Placed after the print statement)
                if any(s in appended_chunk for s in stop_sequences):
                    print("\n🛑 Stop sequence encountered. Ending generation.")
                    break
                    
                # Using 'not correction_token' catches None, "", and [] automatically
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