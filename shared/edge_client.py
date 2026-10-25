import openai
import requests

# Point to your local draft model
local_draft_client = openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
# Point to your port-forwarded HPRC FastAPI server
CLOUD_FASTAPI_URL = "http://127.0.0.1:8001/verify" 

def run_distributed_speculative(prompt):
    current_text = prompt
    
    for step in range(5): # Run 5 rolling steps
        # 1. Draft locally
        draft_resp = local_draft_client.completions.create(
            model="./local_draft_model",
            prompt=current_text,
            max_tokens=5,
            logprobs=1
        )
        draft_tokens = draft_resp.choices[0].logprobs.tokens
        
        # 2. Send the tiny payload to the Cloud Grader via FastAPI
        payload = {
            "prompt": current_text,
            "draft_tokens": draft_tokens
        }
        cloud_grade = requests.post(CLOUD_FASTAPI_URL, json=payload).json()
        
        # 3. Apply the results
        accepted = cloud_grade["accepted_count"]
        correction = cloud_grade["correction_token"]
        
        appended_chunk = "".join(draft_tokens[:accepted])
        if correction:
            appended_chunk += correction
            
        current_text += appended_chunk
        print(f"Step {step+1}: Accepted {accepted} drafts. Appended: {repr(appended_chunk)}")

run_distributed_speculative("The biggest challenge in distributed systems is")