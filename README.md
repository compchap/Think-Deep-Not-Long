# 🧠 Reasoning DTR: Probing Internal LLM Effort
> An implementation of the "Think Deep, Not Just Long" (Google 2026) research paper.

This repository provides a framework for measuring the **Deep-Thinking Ratio (DTR)** in reasoning models (like DeepSeek-R1 and Qwen-Math) by analyzing internal layer convergence.

---

## 🔬 Scientific Context
Traditional reasoning metrics rely on output length (Chain-of-Thought). This project implements **DTR**, a mechanistic metric that identifies tokens requiring sustained revision in deep layers.
- **Hypothesis:** High DTR correlates with accuracy; high length without DTR signals "overthinking." Reasoning Vs Rambling
- **Implementation:** Leverages the **Logit Lens** to probe hidden states $\{h_{t,l}\}$ across 32+ layers.

---

## 🛠️ Roadmap & Implementation Status
- [x] **Algorithm 1:** Deep-Thinking Ratio calculation (JSD-based stabilization).
- [ ] **Algorithm 2:** Think@n Inference Scaling (Early rejection of shallow branches).
- [ ] **Experiment 1:** Correlation Analysis (DTR vs. Accuracy on AIME 2024).
- [ ] **Experiment 2:** Compute-Efficiency Benchmarking (Think@n vs. Self-Consistency).

---

## 🚀 Quick Start (Mac M1/MPS)
### 1. Environment Setup
We use `uv` for deterministic dependency management.

Install [uv](https://github.com/astral-sh/uv).

**Download Dependencies:** `uv sync`


## Setup
- Sync dependencies: `uv pip install -r requirements.txt`.
- Download model: `uv run download_model.py`.

## Running
- **Interactive Test:** `uv run main.py --max-tokens 50 --model deepseek`
- **Benchmark:** `uv run benchmark.py`

## Concept
DTR measures the internal "effort" of a model. If a token's prediction only stabilizes in late layers (High JSD with the final layer), it is a **Deep-Thinking Token**.

## License

MIT License - Copyright (c) 2024-2026 Nikhil Kumar Gupta