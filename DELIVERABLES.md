# Deliverables & Timeline

> **How to read this file.** This is the PMs' best current estimate of what the project needs to ship and roughly when. It is a **living plan, not a contract** — deliverables will be added, dropped, split, or resequenced as the team learns more. The authoritative, up-to-the-minute picture always lives in the repo's **GitHub Issues and Project board**; this file is the high-level narrative that keeps everyone oriented.
>
> PMs: replace the placeholder rows below with your real deliverables. Keep each deliverable small enough to become one or a few issues.

## Milestones at a glance

> Draft milestones — PMs to confirm dates.

| # | Deliverable | Description | Owner (team) | Target |
|---|-------------|-------------|--------------|--------|
| 1 | Onboarding | Read [`docs/READING.md`](docs/READING.md); local vLLM + HPRC access working. | All | Week 2 |
| 2 | Single-node baseline | Speculative decoding (draft + target) on one machine; baseline metrics. | Edge + Cloud | Week 4 |
| 3 | Distributed RPC | Edge draft model ↔ cloud verifier over RPC; schema in [`shared/`](shared/). | Edge + Cloud | Week 7 |
| 4 | TSLT | Sparse logits transmission to cut uplink. | Edge + Cloud | Week 10 |
| 5 | Live dashboard | Streamlit view of TTFT, TPOT, acceptance rate, uplink bytes. | UI | Week 11 |
| 6 | Evaluation | Prove latency reduction vs. cloud-only and edge-only baselines. | Eval | Week 13 |
| 7 | Deployment & demo | Distributed vLLM server on HPRC + dashboard demo. | All | Week 14 |

Week 1: Google Form Quiz: https://forms.gle/kNXyt6A6Vfj15KBV7
Reflection Form: https://forms.gle/51cFMPnyJHqTTkjd7
## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the summary.
- **Dates are estimates.** When reality diverges, update the issue and, if the shift is material, this file. Don't let this file quietly go stale.
- **"Done" is defined per issue** via acceptance criteria — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.

## AccelLLM: Project Roadmap & Operations

Lead Technical Architect: Junyu

Technical Program Manager: Dhruv

Principal Faculty Advisor: Dr. Ihong Hou

1. **Subteam Workflow & Semester Progression**
The AccelLLM project operates across a 16-person roster divided into four highly specialized subteams. Because this architecture involves distributed computing across hardware-constrained local devices and High-Performance Research Computing (HPRC) clusters, strict boundaries and data contracts are required.

Subteam Responsibilities & Execution Plan:
Team 1: Edge & Inference Systems (Lead: Johanna)

Focus: Manages the local hardware environment, executing the lightweight draft model to propose speculative tokens. Responsible for optimizing local continuous batching.

Team 2: Cloud Infrastructure & Distributed RPC (Lead: Kevin, Assist: Patrick)

Focus: Manages the HPRC target model deployments (Texas A&M Grace/Faster clusters) using isolated Docker containers. Handles custom network RPCs and multithreaded continuous batching to efficiently utilize the KV cache.   
PDF

Team 3: Real-Time UI & Data Streaming (Lead: Sree)

Focus: Develops the React/Next.js frontend. Manages the Server-Sent Events (SSE) data pipeline to stream live vLLM token telemetry dynamically.

Team 4: Telemetry Evaluation & Project Management (Lead: Sunghyun)

Focus: Establishes mathematical baselines. Evaluates the statistical improvement in Time To First Token (TTFT), Throughput (Tokens/sec), and token acceptance rates.

Semester Roadmap:
Phase 1: Kickoff & "Tracer Bullet" Pipeline (Weeks 1-3): Establish a basic, unoptimized dummy pipeline. Route a text string from the Edge device to the Cloud container via custom RPC, then push dummy latency metrics to the UI dashboard over SSE.

Phase 2: Core Model Integration (Weeks 4-7): Deploy the chosen draft model on the edge devices and the target model on the HPRC cluster. Implement base speculative decoding where the draft model predicts multiple tokens for verification in a single forward pass.

Phase 3: Network & Memory Optimization (Weeks 8-11): Implement strategies like Truncated Sparse Logits Transmission (TSLT) to drastically reduce uplink network overhead between Edge and Cloud. Tune the PagedAttention memory manager to minimize KV cache fragmentation.   
PDF

Phase 4: Telemetry Benchmarking & Polish (Weeks 12-14): Team 4 runs rigorous statistical evaluations on system performance across various batch sizes. Finalize the dashboard and prepare project documentation.

2. **Finalized Data Sources**
Since AccelLLM focuses on LLM infrastructure and inference optimization, our "data sources" consist of the model weights and the standardized benchmarking datasets used to measure system throughput.

Model Repositories (HuggingFace):

Draft Model (Edge): Qwen/Qwen1.5-1.8B or meta-llama/Llama-3-8B-Instruct (Quantized to run on local hardware constraints).

Target Verifier Model (Cloud): meta-llama/Llama-3-70B-Instruct (Hosted on HPRC Grace/Faster).

Benchmarking Datasets:

ShareGPT (General Chat): Used for testing continuous batching with highly variable prompt and completion lengths.

GSM8K (Reasoning): Used to test generation accuracy and ensure our speculative decoding architecture remains mathematically lossless.

Live Telemetry Data: Telemetry metrics generated dynamically by the system, including Time Per Output Token (TPOT), Time to First Token (TTFT), GPU memory usage, and token acceptance rates.

3. **GitHub Structure & Contribution Workflow**
With 16 active members, AccelLLM will use a strict, structured monorepo environment to prevent merge conflicts between the networking backend and frontend UI.

Repository Structure
The main repository will be divided into specific sub-directories matching the team structures:

/edge (Draft model execution, hardware telemetry)

/cloud (Dockerfiles, multithreaded backend logic, RPC routing)

/ui (React/Next.js dashboard, SSE client logic)

/eval (Python benchmarking scripts, data logs)

Branching Strategy
main: The stable, production-ready codebase. Only updated at the end of major sprints.

dev: The active integration branch. All finalized feature branches merge here first to ensure the edge-to-cloud pipeline doesn't break.

Feature Branches: Members must branch off dev using the following naming convention: <team>/<feature-name> (e.g., cloud/rpc-payload-optimization, ui/sse-metric-routing).

Pull Request (PR) & Merging Rules
Strict Commit Naming: All Discord communication and Git commits must use real first and last names.

PR Templates: All Pull Requests must include a description of the changes, linked Jira/Trello tasks, and proof of local testing.

Review Requirements: A PR submitted to dev requires at least one approval from the respective subteam lead (Johanna, Kevin, Sree, or Sunghyun).

Cross-Team Merging: For PRs that bridge multiple systems (e.g., changing the data payload sent from the Cloud to the UI), approval is required from both affected team leads before merging.

Final Deployment: The Technical Program Manager (Dhruv) and Lead Technical Architect (Junyu) will handle merging dev into main after verifying the end-to-end integration.