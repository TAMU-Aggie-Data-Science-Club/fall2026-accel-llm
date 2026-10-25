from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import openai # Using local vLLM instance on the cluster
import uvicorn

app = FastAPI()

# Point to the vLLM instance running on the SAME HPRC cluster (localhost for the cluster)
local_vllm_client = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

@app.websocket("/ws/verify")
async def verify_draft_ws(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Client connected via WebSocket.")
    try:
        # Keep the connection open for the entire generation loop
        while True:
            data = await websocket.receive_json()
            
            # 1. Cloud server queries the massive target model locally (microsecond latency)
            target_resp = local_vllm_client.completions.create(
                model="/scratch/user/junyuxu/my_local_model",
                prompt=data["prompt"],
                max_tokens=len(data["draft_tokens"]),
                logprobs=1
            )
            target_tokens = target_resp.choices[0].logprobs.tokens
            
            # 2. Cloud server performs the accept/reject math right here on the cluster
            accepted_count = 0
            correction = None
            
            for i in range(min(len(data["draft_tokens"]), len(target_tokens))):
                if data["draft_tokens"][i] == target_tokens[i]:
                    accepted_count += 1
                else:
                    correction = target_tokens[i]
                    break
                    
            if accepted_count == len(data["draft_tokens"]) and len(target_tokens) > accepted_count:
                correction = target_tokens[accepted_count]

            # 3. Stream the tiny final grade back down the open SSH tunnel
            await websocket.send_json({
                "accepted_count": accepted_count,
                "correction_token": correction
            })
    except WebSocketDisconnect:
        print("🛑 Client finished generation and disconnected.")

if __name__ == "__main__":
    # Binds to 0.0.0.0 so it can receive routed traffic on the HPRC node
    uvicorn.run(app, host="0.0.0.0", port=8001)