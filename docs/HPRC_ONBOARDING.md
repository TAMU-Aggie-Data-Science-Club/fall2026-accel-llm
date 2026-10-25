# HPRC model run and evidence checklist

This guide prepares the project for the requested onboarding and benchmark
evidence. It does not claim that a model was run or that HPRC resources were
allocated. Run the GPU steps on an allocated compute node, not on a login node.

## Before downloading model weights

1. Get an active TAMU HPRC account and confirm which cluster and allocation
   (FASTER, Grace, or another assigned resource) to use.
2. Confirm the available GPU type/count, memory, partition, account/charge code,
   scratch quota, and the resource request needed for a 70B target. Do not assume
   that a one-GPU allocation can serve the target.
3. Review and accept the model terms under your own account:
   - Draft: [`Qwen/Qwen1.5-1.8B`](https://huggingface.co/Qwen/Qwen1.5-1.8B) is
     published under the Tongyi Qianwen Research License. The license grants
     non-commercial use; confirm the project use complies before downloading.
   - Target: [`meta-llama/Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
     is gated. Request access and accept Meta's Llama 3 Community License on
     Hugging Face before downloading.
4. Use the HPRC-supported Python/CUDA/vLLM environment for the selected cluster.
   Check its current software guide and module catalog first; do not blindly
   install a CUDA/PyTorch/vLLM stack or compile it on a login node.

Official starting points:

- [HPRC account and cluster access](https://hprc.tamu.edu/kb/)
- [FASTER user guide](https://hprc.tamu.edu/kb/User-Guides/FASTER/)
- [FASTER SLURM batch guide](https://hprc.tamu.edu/kb/User-Guides/FASTER/Batch/)
- [Grace user guide](https://hprc.tamu.edu/kb/User-Guides/Grace/)
- [HPRC policies](https://hprc.tamu.edu/policies/)

## Access and interactive GPU allocation

The form asks which access method you used. Select the one you actually use:

- **SSH Terminal:** the official FASTER guide gives
  `ssh <NetID>@faster.hprc.tamu.edu` as the login pattern. Replace the
  placeholder with your NetID; do not share passwords or tokens.
- **TAMU Open OnDemand Portal:** use the portal for the cluster assigned to your
  account. Record the cluster, interactive app/job type, partition, GPU type and
  count, CPUs, memory, and wall time shown in the request form. Portal settings
  are not interchangeable with an SSH command.

For FASTER, the official batch guide documents a `gpu` partition and
`--gres=gpu:1`. This is an example request, not a claim about your allocation
and not a guarantee that a GPU is available or sufficient for the 70B target:

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash -l
```

If your allocation requires an account/charge code, add the account option
provided by HPRC. Check the current cluster guide or ask HPRC which GPU request
is appropriate for the target model before running the target. The exact command
to paste in the form is the one you actually submit; save it verbatim. Do not
report this generic example as a command already used.

Heavy compiling, vLLM serving, and sustained inference belong on allocated
compute/GPU nodes. Base login nodes are shared for lightweight tasks such as
editing, file management, module discovery, and job submission. Running heavy
jobs there can affect other users and bypass scheduler resource controls.

## Capture the allocation and environment evidence

From the allocated interactive worker, save terminal output and capture the
requested screenshots:

```bash
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not-set}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-not-set}"
module list 2>&1
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

Capture the `module list` output as text for the form. The `nvidia-smi`
screenshot must show the worker node, GPU model, and memory. Do not present a
login-node GPU check as worker-node evidence.

Use the cluster's supported Python module/environment, then run this basic GPU
visibility check in the allocated job:

```bash
python -c 'import torch; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

This requires PyTorch to be present in the supported environment. A successful
GPU check proves the Python environment can see CUDA; it does not prove vLLM
initialized or that either model fits.

## Download weights to HPRC scratch

Do not download the 70B target to this Mac. After license/access approval, use
the Hugging Face CLI version supported by the cluster and store files in
HPRC-provided scratch (never in Git):

```bash
mkdir -p "$SCRATCH/models"
hf auth login
hf download Qwen/Qwen1.5-1.8B --local-dir "$SCRATCH/models/Qwen1.5-1.8B"
hf download meta-llama/Meta-Llama-3-70B-Instruct \
  --local-dir "$SCRATCH/models/Meta-Llama-3-70B-Instruct"
```

`hf auth login` is interactive and uses a Hugging Face access token. Never put
the token in a script, command history, screenshot, or uploaded log. If the
cluster only provides the older `huggingface-cli` command, follow that
environment's current Hugging Face instructions instead. Verify model files
and available scratch space before starting a download. Large transfers and
installations should follow the cluster's current usage policy.

## Start services and run a smoke test

The project currently has endpoint/client scaffolding, not a verified
end-to-end vLLM deployment recipe. First load the documented cluster-supported
environment and confirm vLLM compatibility for that cluster's CUDA/PyTorch
stack. Then configure the draft and target servers with the actual model paths,
GPU allocation, and distinct ports. Do not copy the hard-coded personal
`/scratch/...` path in `shared/cloud_server.py`.

Once a server is running, record an actual prompt, response, and wall-clock
latency. For example, with a compatible OpenAI-style server on port 8000:

```bash
date
/usr/bin/time -p curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-model-id>","prompt":"Write one sentence explaining why context length uses memory.","max_tokens":40,"temperature":0}' \
  | tee prompt-response.json
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

Replace `<served-model-id>` with the ID returned by the server's `/v1/models`.
Keep the actual terminal output showing the input prompt, generated text,
`real` elapsed time, and GPU memory reading. The GPU-wide memory reading may
include other processes; label it as device usage rather than attributing it
all to your process.

To study context length, repeat the same prompt/task at controlled input lengths
and output lengths, recording GPU memory before and during/after each run. KV
cache use grows with active tokens and concurrent sequences, so longer contexts
reduce the number of requests that fit on a memory-limited edge GPU. That
capacity tradeoff is a major continuous-batching bottleneck: fewer sequences fit
at once, or requests must wait/preempt, reducing throughput.

## Benchmark and logprob evidence

The requested target-alone versus speculative-decoding screenshot must come
from two completed runs with the same prompt, target model, output-token limit,
generation settings, and allocated hardware. Save the actual total wall-clock
times and generated output for both. Do not substitute a demo, a model-loading
time, or an unverified estimate.

**Important:** `shared/distributed_benchmark.py` is not currently validated as
a correct speculative-decoding benchmark. In particular, the cloud verifier
uses a hard-coded model path and doesn't set the target sampling parameters to
match the benchmark; the code compares greedy token strings rather than
implementing the paper's sparse-logit accept/reject procedure. Its timing
should not be submitted as proof of TSLT or valid speculative decoding until
the model path, matched settings, output lengths, and verification algorithm
are corrected and checked.

For a brief top-five logprob capture, vLLM's compatible completions API accepts
a `logprobs` count. With the project's `openai` Python client installed and a
server listening on port 8000, run:

```bash
python - <<'PY'
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
model = client.models.list().data[0].id
result = client.completions.create(
    model=model,
    prompt="The capital of Texas is",
    max_tokens=1,
    temperature=0,
    logprobs=5,
)
logprobs = result.choices[0].logprobs
print("prompt:", "The capital of Texas is")
print("generated token:", logprobs.tokens[0])
print("top logprobs:", logprobs.top_logprobs[0])
PY
```

The exact response schema depends on the installed API/client version. Capture
real values from a run; do not invent values or include credentials in the
screenshot.

When a draft is inaccurate, fewer proposed tokens are accepted. Rejected work
still incurs draft, verifier, and communication overhead, so the combined
pipeline may be no faster—or slower—than target-only generation.

For TSLT, the verifier needs the sparse token/log-probability information
required by the protocol's sampling and verification procedure, not only the
final decoded text. Text alone loses the draft distribution and uncertainty;
the verifier cannot reconstruct the probability mass needed to evaluate
proposals or preserve the intended sampling distribution. Transmitting selected
probabilities/token IDs instead of a full vocabulary distribution reduces
communication while retaining the information required by the protocol.

## Form evidence checklist

| Form item | Evidence to collect |
|---|---|
| Model run locally | Exact model repository/path and precision or quantization actually served. |
| Prompt, response, latency, local memory/VRAM | Terminal output from one real generation plus wall time and a labeled memory reading. |
| Context length and continuous batching | Explanation above, supported by recorded memory at controlled context lengths if requested as a local result. |
| HPRC access method | SSH or OnDemand, whichever was actually used. |
| Exact interactive request | Verbatim submitted `srun`/`salloc` command or actual portal settings. |
| Why not use login node | Shared-resource and scheduler explanation above. |
| `module list` | Text output captured inside the active environment/allocation. |
| Worker `nvidia-smi` | Screenshot from the allocated worker showing GPU model and memory. |
| Python GPU check or vLLM log | Actual command output from the allocated worker. |
| Target-only vs. speculative timing | Screenshot of matched completed runs after validating the speculative verifier. |
| Low draft acceptance | Explanation above; actual speed claim requires measurements. |
| Top 3–5 logprobs | Real one-token response with `logprobs` set to 3–5. |
| Why transmit logprobs (TSLT) | Explanation above, grounded in the [TSLT reading](https://arxiv.org/abs/2512.16273). |
