# Graph-R1: Agentic GraphRAG via End-to-end Reinforcement Learning

A clean reproduction of the **Graph-R1** paper (ICML 2026) — the first agentic GraphRAG framework powered by end-to-end reinforcement learning using GRPO.

> **Paper**: [Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](https://arxiv.org/abs/2507.21892)
> **Original Code**: [LHRLAB/Graph-R1](https://github.com/LHRLAB/Graph-R1)

## Overview

Graph-R1 addresses three key challenges in existing GraphRAG methods:
1. **High construction cost** — uses lightweight knowledge hypergraph construction
2. **Fixed one-time retrieval** — models retrieval as multi-turn agent-environment interaction
3. **Dependence on large LLMs** — optimizes small models via end-to-end RL (GRPO)

The agent learns to iteratively **think → query → retrieve → rethink** over a knowledge hypergraph, with its policy optimized by outcome-directed rewards.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Knowledge Hypergraph G_H         │
                    │    (V = entities, E_H = hyperedges, φ)   │
                    │                                           │
                    │  Documents → Chunks → N-ary Relations     │
                    │  → Entity/Hyperedge Embeddings (BGE)      │
                    │  → FAISS Indices                          │
                    └────────────────┬────────────────────────┘
                                     │
    ┌────────────────────────────────┼────────────────────────────────┐
    │                   Multi-turn Agent Loop                         │
    │                                                                 │
    │   Question → <think> reasoning </think>                         │
    │            → <query> retrieval query </query>                   │
    │            → [Entity Retrieval + Hyperedge Retrieval]           │
    │            → [Reciprocal Rank Fusion]                           │
    │            → <knowledge> retrieved facts </knowledge>           │
    │            → ... (repeat up to T turns) ...                     │
    │            → <think> final reasoning </think>                   │
    │            → <answer> final answer </answer>                    │
    │                                                                 │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────┼────────────────────────────────┐
    │              GRPO Training (End-to-end RL)                      │
    │                                                                 │
    │   R(τ) = -1.0 + R_format(τ) + 𝟙{R_format=1} · R_answer(F1)   │
    │   Advantage: Â(τ) = (R(τ) - mean) / std                       │
    │   Loss: Clipped PPO + β · KL(π_θ ‖ π_ref)                     │
    └─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Graph-R1-Paper-Implementation/
├── src/graph_r1/              # Core implementation
│   ├── hypergraph.py          # Knowledge Hypergraph Construction (Sec 4.1)
│   ├── retrieval.py           # Dual-path retrieval + RRF (Sec 4.2)
│   ├── agent.py               # Multi-turn agentic reasoning (Sec 4.2)
│   ├── rewards.py             # Format + Answer rewards (Sec 4.3)
│   ├── grpo_trainer.py        # GRPO training pipeline (Sec 4.3)
│   ├── evaluate.py            # EM, F1, R-S evaluation metrics
│   └── data.py                # FlashRAG dataset loading
├── configs/                   # Training configurations
│   ├── grpo_1.5b.json         # Qwen2.5-1.5B config
│   └── grpo_3b.json           # Qwen2.5-3B config
├── scripts/                   # Utility scripts
│   ├── prepare_data.py        # Download & format datasets
│   ├── build_hypergraph.py    # Build knowledge hypergraph
│   ├── test_local.py          # Local validation tests
│   └── run_kaggle.sh          # Kaggle experiment launcher
├── notebooks/                 # Kaggle experiment notebooks
│   └── graph_r1_experiment.py # Full experiment pipeline
├── experiments/               # Experiment logs and results
│   └── experiment_log.md      # Experiment tracking
└── 2507.21892v2.pdf           # Original paper
```

## Method Details

### 1. Knowledge Hypergraph Construction (Section 4.1)

- **Chunking**: Documents split into 1200-token windows with 50-token overlap
- **N-ary Relation Extraction**: LLM extracts semantic segments (hyperedges) and entities from each chunk
- **Embedding**: Entities and hyperedges encoded with `bge-large-en-v1.5`
- **Indexing**: FAISS inner-product indices for fast similarity search

### 2. Multi-turn Agentic Reasoning (Section 4.2)

The agent has four action types per step:
- **Think** (`<think>...</think>`): Reflect on current state, identify knowledge gaps
- **Query** (`<query>...</query>`): Generate a retrieval query
- **Retrieve**: Dual-path retrieval from hypergraph
  - Entity-based: Find similar entities → collect connected hyperedges
  - Direct: Find similar hyperedges directly
  - Fusion: Reciprocal Rank Aggregation (Score = 1/r_V + 1/r_H)
- **Answer** (`<answer>...</answer>`): Produce final response

### 3. GRPO Training (Section 4.3)

- **Rollout Generation**: Sample N=5 trajectories per question
- **Reward**: R(τ) = -1.0 + R_format + 𝟙{R_format ≥ threshold} · R_answer(F1)
  - Format reward: 0.5 per valid think/query or think/answer block (capped at 1.0)
  - Answer reward: Token-level F1 score vs ground truth
  - Threshold: 1.0 in paper (multi-turn required), 0.5 for LoRA adaptation
- **Advantage**: Group-normalized within same-prompt rollouts
- **Policy Update**: Clipped PPO loss with KL regularization (β=0.001)

## Setup

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Kaggle account (for running experiments)

### Installation

```bash
git clone https://github.com/jivansh77/Graph-R1-Paper-Implementation.git
cd Graph-R1-Paper-Implementation
pip install -r requirements.txt
```

### Local Validation

```bash
python scripts/test_local.py
```

## Running Experiments

### On Kaggle (Recommended)

1. Set up Kaggle API credentials:
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
```

2. Push and run the experiment:
```bash
cd notebooks/
kaggle kernels push
```

3. Monitor progress:
```bash
kaggle kernels status jivanshc/graph-r1-reproduction-2wikimultihopqa
```

4. Download results:
```bash
kaggle kernels output jivanshc/graph-r1-reproduction-2wikimultihopqa -p experiments/
```

### Local Execution

1. Prepare data:
```bash
python scripts/prepare_data.py --dataset 2WikiMultiHopQA --data_dir data/
```

2. Build hypergraph:
```bash
python scripts/build_hypergraph.py --dataset 2WikiMultiHopQA --data_dir data/
```

3. Train (requires GPU):
```python
from src.graph_r1.grpo_trainer import GRPOConfig, GRPOTrainer
config = GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
trainer = GRPOTrainer(config=config, retriever=retriever)
trainer.train(train_data, eval_data=test_data)
```

## Datasets

Six standard RAG benchmarks from FlashRAG (5,120 train / 128 test each):

| Dataset | Type | Description |
|---------|------|-------------|
| 2WikiMultiHopQA | Multi-hop | Reasoning across two Wikipedia documents |
| HotpotQA | Multi-hop | Diverse question types with supporting facts |
| Musique | Multi-hop | Chains of 3+ reasoning steps |
| Natural Questions | Single-hop | Real Google search questions |
| PopQA | Single-hop | Popular culture questions from Wikipedia |
| TriviaQA | Single-hop | Trivia-style distantly supervised QA |

## Paper Results (Table 2)

### Qwen2.5-1.5B-Instruct

| Method | 2Wiki F1 | HotpotQA F1 | Musique F1 | NQ F1 | PopQA F1 | TriviaQA F1 | Avg F1 |
|--------|----------|-------------|------------|-------|----------|-------------|--------|
| NaiveGeneration | 49.13 | 45.77 | 2.35 | 46.74 | 42.67 | 52.92 | 47.31 |
| StandardRAG | 55.38 | 52.91 | 3.18 | 59.73 | 50.29 | 60.52 | 53.05 |
| Search-R1 | 58.81 | 61.54 | 6.26 | 36.86 | 38.37 | 61.24 | 56.12 |
| **Graph-R1 (paper)** | **65.73** | **65.30** | **28.28** | **59.13** | **66.46** | **70.83** | **64.38** |

### Qwen2.5-3B-Instruct

| Method | 2Wiki F1 | HotpotQA F1 | NQ F1 | PopQA F1 | TriviaQA F1 | Avg F1 |
|--------|----------|-------------|-------|----------|-------------|--------|
| **Graph-R1 (paper)** | **76.45** | **77.46** | **69.42** | **71.27** | **75.01** | **72.99** |

## Reproduction Results

### 2WikiMultiHopQA (Qwen2.5-1.5B-Instruct)

| Metric | Paper | Ours | Gap |
|--------|-------|------|-----|
| EM     | 35.13 | TBD  | —   |
| F1     | 65.73 | TBD  | —   |

*Results pending — v9 run in progress (2026-08-28). See `experiments/experiment_log.md` for full run history.*

### Training Observations

Over 9 iterations, key findings on adapting Graph-R1 for single-GPU LoRA training:

1. **Learning rate matters most**: The paper's LR (5e-7) is tuned for full-parameter training. LoRA updates ~0.5% of parameters and needs ~40x higher LR (2e-5) to produce meaningful parameter updates. With the paper's LR, per-step losses were ~1e-7 — effectively zero learning.

2. **Format reward gating is too strict under compute constraints**: The paper gates answer reward behind `R_format = 1.0`, requiring multi-turn `<think>`+`<query>` followed by `<think>`+`<answer>`. A single-turn response with perfect think+answer structure only earns 0.5 format reward, so the model must discover multi-turn retrieval patterns before getting any answer feedback. We lower the threshold to 0.5, letting single-turn format unlock answer credit while still incentivizing multi-turn through higher format scores.

3. **Per-rollout backward prevents OOM**: Accumulating loss tensors across all rollouts keeps full computational graphs in memory. Processing each rollout's backward pass independently and freeing the graph after each prevents the P100's 16GB from being exhausted.

4. **Gradient checkpointing is essential**: Enables 1.5B model training on 16GB VRAM at the cost of ~30% slower forward passes.

## Differences from Original

| Aspect | Original | Our Reproduction |
|--------|----------|-----------------|
| Compute | 4× A100 80GB | Kaggle T4/P100 16GB |
| Training | Full parameters via VERL/Ray | LoRA (r=16) via PyTorch |
| Batch size | 128 | 4 (with gradient accumulation) |
| Rollouts | 5 per prompt | 3 per prompt |
| Learning rate | 5e-7 (full-param) | 2e-5 (LoRA) + cosine decay |
| Format threshold | 1.0 (multi-turn required) | 0.5 (single-turn unlocks F1) |
| N-ary extraction | GPT-4o-mini API | Local regex-based |
| Framework | VERL + Ray distributed | Custom single-GPU PyTorch |

These differences primarily affect training scale and graph construction quality, but the core algorithm (GRPO with format+answer rewards over hypergraph retrieval) is faithfully implemented.

## Evaluation Metrics

- **EM** (Exact Match): Binary match after normalization
- **F1**: Token-level precision/recall harmonic mean
- **R-S** (Retrieval Similarity): Cosine similarity of retrieved vs gold knowledge embeddings
- **G-E** (Generation Evaluation): 7-dimension quality score via LLM judge

## Citation

```bibtex
@inproceedings{luo2026graphr1,
  title={Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning},
  author={Luo, Haoran and E, Haihong and Chen, Guanting and Lin, Qika and Guo, Yikai and Xu, Fangzhi and Kuang, Zemin and Song, Meina and Wu, Xiaobao and Zhu, Yifan and Tuan, Luu Anh},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## License

This reproduction is for research and educational purposes. The original Graph-R1 code is released under the MIT License.
