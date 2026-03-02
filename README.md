# Reasoning DTR: Probing Internal LLM Effort

> An implementation of the "Think Deep, Not Just Long" (Google 2026) research paper.

This repository provides a framework for measuring the **Deep-Thinking Ratio (DTR)** in reasoning models (like DeepSeek-R1 and Qwen-Math) by analyzing internal layer convergence. I have also implemented the **Think@n**, a test-time scaling strategy that uses DTR to select high-quality samples while reducing inference costs by ~50%

---

## Scientific Context

Traditional reasoning metrics rely on output length (Chain-of-Thought). This project implements **DTR**, a mechanistic metric that identifies tokens requiring sustained revision in deep layers.

- **Hypothesis:** High DTR correlates with accuracy; high length without DTR signals "overthinking." Reasoning Vs Rambling
- **Implementation:** Leverages the **Logit Lens** to probe hidden states $h_{t,l}$ across 32+ layers.

---

## Roadmap & Implementation Status

- **Algorithm 1:** Deep-Thinking Ratio calculation (JSD-based stabilization).
- **Algorithm 2:** Think@n Inference Scaling (Early rejection of shallow branches).
- **Experiment 1:** Correlation Analysis (DTR vs. Accuracy on AIME 2024).
- **Experiment 2:** Compute-Efficiency Benchmarking (Think@n vs. Self-Consistency).

---

## Quick Start

### 1. Environment Setup

We use `uv` for deterministic dependency management.

Install [uv](https://github.com/astral-sh/uv).

**Download Dependencies:** `uv sync`

## Setup

- Sync dependencies: `uv pip install -r requirements.txt`.
- Download model: `uv run download_model.py`.

## Running

- **Algorithm 1 (DTR):** `uv run examples/demo_dtr.py --max-tokens 50 --model qwen.6b`
- **Algorithm 2 (Think@n):** `uv run examples/demo_think_at_n.py --model qwen.6b --n 12`
- **Benchmark:** `uv run benchmark.py`

## Usage Examples

### Algorithm 1: Deep-Thinking Ratio (Single Sample)

```python
from src.dtr_engine import DTREngine

# Initialize engine
engine = DTREngine(model_id="Qwen/Qwen3-0.6B", cache_dir="models/qwen")

# Generate with DTR tracking
for token, is_deep, dtr in engine.generate_with_dtr("Calculate 12 * 12: ", max_tokens=50):
    marker = "🧠" if is_deep else "  "
    print(f"{marker} '{token}' | DTR: {dtr:.2f}")
```

### Algorithm 2: Think@n Inference Scaling (Multi-Sample)

```python
from src.dtr_engine import DTREngine
from src.think_at_n import ThinkAtN

# Initialize
engine = DTREngine(model_id="Qwen/Qwen3-0.6B", cache_dir="models/qwen")
think = ThinkAtN(engine, n=12, eta=0.5, prefix_length=50)

# Solve with multiple methods
result = think.solve(
    problem="Calculate 12 * 12",
    ground_truth="144"
)

print(f"Think@n answer: {result['think_at_n']['answer']}")
print(f"Cost savings: {(1 - result['think_at_n']['cost'] / result['cons_at_n']['cost']) * 100:.1f}%")
```

**Key Features:**

- **Early Stopping**: Generates prefix (50 tokens), ranks by DTR, continues only top 50%
- **Cost Reduction**: ~50% token savings vs self-consistency
- **Accuracy**: Matches/exceeds Cons@n by selecting high-DTR samples
- **Baselines**: Compares against Cons@n, Short@n, Long@n

## Hardware

 - Mac M1

## Concept

DTR measures the internal "effort" of a model. If a token's prediction only stabilizes in late layers (High JSD with the final layer), it is a **Deep-Thinking Token**.

## License

MIT License - Copyright (c) 2024-2026 Nikhil Kumar Gupta