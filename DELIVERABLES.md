# Deliverables & Timeline

> **How to read this file.** This is the PMs' best current estimate of what the project needs to ship and roughly when. It is a **living plan, not a contract** — deliverables will be added, dropped, split, or resequenced as the team learns more. The authoritative, up-to-the-minute picture always lives in the repo's **GitHub Issues and Project board**; this file is the high-level narrative that keeps everyone oriented.
>
> PMs: replace the placeholder rows below with your real deliverables. Keep each deliverable small enough to become one or a few issues.

## Milestones at a glance

> Draft milestones — PMs to confirm dates.

| # | Deliverable | Description | Owner (team) | Target |
|---|-------------|-------------|--------------|--------|
| 1 | Onboarding | Read [`docs/READING.md`](docs/READING.md); local vLLM + HPRC access working. | All | TBD |
| 2 | Single-node baseline | Speculative decoding (draft + target) on one machine; baseline metrics. | Edge + Cloud | TBD |
| 3 | Distributed RPC | Edge draft model ↔ cloud verifier over RPC; schema in [`shared/`](shared/). | Edge + Cloud | TBD |
| 4 | TSLT | Sparse logits transmission to cut uplink. | Edge + Cloud | TBD |
| 5 | Live dashboard | Streamlit view of TTFT, TPOT, acceptance rate, uplink bytes. | UI | TBD |
| 6 | Evaluation | Prove latency reduction vs. cloud-only and edge-only baselines. | Eval | TBD |
| 7 | Deployment & demo | Distributed vLLM server on HPRC + dashboard demo. | All | TBD |

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the summary.
- **Dates are estimates.** When reality diverges, update the issue and, if the shift is material, this file. Don't let this file quietly go stale.
- **"Done" is defined per issue** via acceptance criteria — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.
