import math
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import openai
import uvicorn

app = FastAPI()

# Point to the vLLM instance running on the SAME HPRC cluster
local_vllm_client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
MODEL_PATH = "/scratch/user/junyuxu/my_local_model"

@app.websocket("/ws/verify")
async def verify_draft_ws(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Client connected via WebSocket (Single-Pass Parallel Verification).")
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data["prompt"]
            draft_tokens = data["draft_tokens"]
            draft_logprobs = data.get("draft_logprobs", [0.0] * len(draft_tokens))
            k = len(draft_tokens)

            if k == 0:
                await websocket.send_json({"accepted_count": 0, "correction_token": None})
                continue

            # 1. Concatenate prompt + draft tokens to evaluate everything in 1 parallel pass
            combined_prompt = prompt + "".join(draft_tokens)

            target_resp = local_vllm_client.completions.create(
                model=MODEL_PATH,
                prompt=combined_prompt,
                max_tokens=1,       # Only 1 forward pass! Generates the (k+1)th bonus token
                temperature=0.0,
                logprobs=1,
                echo=True,          # Echoes prompt tokens so we can inspect prompt_logprobs
                extra_body={"prompt_logprobs": 1}
            )

            choice = target_resp.choices[0]
            all_tokens = choice.logprobs.tokens
            all_top_logprobs = choice.logprobs.top_logprobs

            # Slice out the k drafted positions and the 1 newly generated bonus token at the end
            # all_tokens[-1] is the generated bonus token; all_tokens[-(k+1):-1] are the k draft tokens
            draft_Slice_top_logprobs = all_top_logprobs[-(k + 1):-1]
            bonus_token = all_tokens[-1]

            accepted_count = 0
            correction = None

            # 2. Parallel Rejection Sampling: accept with probability min(1, p(x) / q(x))
            for i in range(k):
                d_tok = draft_tokens[i]
                q_logprob = draft_logprobs[i]
                q_prob = math.exp(q_logprob) if q_logprob is not None else 1.0

                # Extract Target's top predicted token and logprob at position i
                pos_logprobs_dict = draft_Slice_top_logprobs[i] or {}
                
                if d_tok in pos_logprobs_dict:
                    # Draft token matches Target's top-1 token (or is in returned top logprobs)
                    p_logprob = pos_logprobs_dict[d_tok]
                    p_prob = math.exp(p_logprob)
                else:
                    p_prob = 0.0

                # Acceptance test: min(1, p(x) / q(x))
                accept_prob = min(1.0, p_prob / max(q_prob, 1e-12))
                
                if random.random() < accept_prob:
                    accepted_count += 1
                else:
                    # Rejected! Pick the target model's top token at this exact position as correction
                    if pos_logprobs_dict:
                        correction = next(iter(pos_logprobs_dict.keys()))
                    break

            # 3. If all k draft tokens were accepted, we get the (k+1)th bonus token for free!
            if accepted_count == k and bonus_token:
                correction = bonus_token

            await websocket.send_json({
                "accepted_count": accepted_count,
                "correction_token": correction
            })

    except WebSocketDisconnect:
        print("🛑 Client finished generation and disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)