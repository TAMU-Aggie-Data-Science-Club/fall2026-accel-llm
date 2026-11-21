# AccelLLM: Distributed Speculative Decoding from Edge to Cloud
Junyu Xu · Dhruv Raajeev

## Introduction
LLMs write one token at a time, and each token needs a full pass through the model.
- **Laptop model:** private and fast, but less capable.
- **Cloud model:** far more capable, but slow and costly to reach.
**Speculative decoding** gets both: a small *draft* model guesses *k* tokens and a large *target* model checks them in one pass, so the output stays the target's.
**α (acceptance rate):** the share of drafted tokens kept.

## Objective
Can a draft model on a laptop and a target model on TAMU's HPRC cluster, linked over the network, generate text faster than the cloud model alone, with no loss in quality?

## Methods
1. **Draft:** the laptop model proposes *k* tokens with probabilities q(x).
2. **Send:** a persistent WebSocket (SSH tunnel) carries them to a FastAPI verifier on an HPRC GPU node (Figure 1).
3. **Verify:** the target scores all *k* in one pass and keeps each with probability min(1, p/q).
4. **Benchmark:** a Streamlit dashboard times cloud-only vs. distributed; k = 3, 5, 8 on 6 prompt types.
**Tools:** vLLM, FastAPI, Slurm on HPRC, Streamlit. **Team:** 16 members in 4 sub-teams.

## Results
The pipeline runs end to end, laptop to HPRC, with every token checked by the target model.
- **Figure 2:** distributed took **3.9 s and 3.7 s** vs. **1.3 s and 0.8 s** cloud-only (0.21–0.34×). The best dashboard run reached 0.38×.
- **Figure 3:** acceptance tracks predictability: code 80%, factual 34%, creative 19%.
- Drafting more tokens per round only pays off when α is high.

## Discussion
- **Why it's slower:** every round crosses laptop → VPN → SSH → GPU node, while cloud-only pays the network once per reply. The round trip costs more than the extra tokens save (Figure 2).
- **Where it wins:** long, predictable outputs (code, JSON, summaries) at low temperature, on a fast network (Figure 3).
- **Next steps:** run on the campus network straight to the GPU node, a better-matched draft model to raise α, TSLT to shrink each message, and drafting the next round during verification.

## Conclusion
Distributed speculative decoding keeps the cloud model's quality by design, but its speed depends on two numbers: **acceptance rate** and **network round-trip time**. The next gains come from cutting the round trip, not from bigger models.

## References
1. Leviathan et al. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
2. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv 2023.
3. Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM). SOSP 2023.
4. Zheng et al. *Fast Collaborative Inference via Distributed Speculative Decoding* (TSLT). arXiv:2512.16273, 2025.
5. github.com/TAMU-Aggie-Data-Science-Club/fall2026-accel-llm

## Contact
Scan for the demo video and code. **[QR code: Cyrus's demo video link]**
