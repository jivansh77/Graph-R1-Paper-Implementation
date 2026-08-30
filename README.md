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
│   └── grpo_3b.json          # Qwen2.5-3B config
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

| Metric | Paper | Ours (v20) | Gap |
|--------|-------|------------|-----|
| EM     | 35.13 | 10.94      | -24.19 |
| F1     | 65.73 | 21.09      | -44.64 |

**Training config**: 512 train samples, batch_size=4, mini_batch_size=2, num_rollouts=3, LR=2e-5, LoRA r=16, temperature=1.0, 1 epoch on Kaggle P100 16GB (~6.1 hours).

**Hypergraph**: 15,792 entities, 9,747 hyperedges, avg 4.4 entities per hyperedge.

The gap vs paper results is expected given the significant compute and methodology differences (see [Differences from Original](#differences-from-original)). This reproduction validates that the core algorithm works — the model learns to use `<query>` tags for retrieval and produces meaningful answers — while identifying the specific challenges of adapting distributed full-parameter GRPO training to single-GPU LoRA fine-tuning.

### Training Dynamics (v20)

- Rewards showed healthy diversity: max reward reached 1.0, mean reward improved over training
- Eval F1 improved over time (13.1% at step 64 → 29.0% at step 192)
- Model successfully learned the think → query → retrieve → answer loop
- SFT warmup (30 steps) was essential for bootstrapping the multi-turn format

## The 20-Run Journey

Getting from 0% to 10.9% EM / 21.1% F1 required 20 Kaggle kernel iterations, uncovering and fixing layered bugs that only became visible once earlier ones were resolved:

### Critical Bugs Discovered

1. **Answer extraction bug (runs 8-19)**: The instruction template contained `<answer>...</answer>` as placeholder text. `extract_answer()` used `re.search` (first match) on the full prompt+response, always matching the instruction placeholder instead of the model's actual answer — returning literal `"..."`. This caused **0% F1 in every single run** because `F1("...", gold_answer) = 0` always. During training, the model never received credit for correct answers. Fixed by switching to `re.findall` + taking the last match.

2. **Float16 NaN loss (run 18)**: The P100 GPU uses float16 (max value 65504), while the paper's A100s use bfloat16. Operations like `log_softmax` and `exp(ratio)` in `compute_policy_loss` overflowed float16, producing NaN gradients that corrupted all model weights within 10 steps. Fixed by casting logits to float32, adding `GradScaler`, and clamping policy ratios.

3. **Advantage collapse (run 19)**: With temperature=0.7 and 3 rollouts, all rollouts quickly converged to the same reward. GRPO's group normalization then produced `(0-0)/(0+ε) ≈ 0` — zero gradients, zero learning. The model found a degenerate local optimum: `<answer>...</answer>` (literal ellipsis). Fixed by skipping normalization when std < 0.01, increasing temperature to 1.0, and penalizing degenerate answers.

4. **P100 CUDA incompatibility (run 1)**: PyTorch 2.10.0+cu128 requires CUDA capability ≥ 7.0; the P100 has 6.0. Required downgrading to torch 2.4.0+cu121.

5. **GPU OOM (run 6)**: Accumulating all rollout loss tensors kept full computational graphs in memory. Fixed by per-rollout backward passes with graph freeing after each.

### Run History Summary

| Run | Hardware | Result | Issue |
|-----|----------|--------|-------|
| 1-5 | GPU | Failed | CUDA compat, imports, torchvision, transformers, torchao |
| 6 | GPU | OOM at step 85 | Accumulated computational graphs |
| 8 | GPU | EM=0%, F1=0% | LR too low (5e-7), format reward too strict |
| 9 | GPU | EM=0%, F1=0% | Model learned format but never used `<query>` tags |
| 10-12 | TPU | Failed | torch_xla gradient checkpointing incompatible |
| 13 | TPU | Killed after 5.7h | XLA autoregressive generation too slow |
| 14-15 | TPU | OOM | Model + ref_model exceeds 15.75GB TPU HBM |
| 16 | TPU | Killed after 4.9h | CPU generation too slow (35h projected) |
| 17 | GPU | NaN crash | Float16 overflow in generation logits |
| 18 | GPU | EM=0%, F1=0% | NaN in policy loss (float16 overflow) |
| 19 | GPU | EM=0%, F1=0% | Advantage collapse + answer extraction bug |
| **20** | **GPU** | **EM=10.9%, F1=21.1%** | **All fixes applied** |

## Differences from Original

| Aspect | Original | Our Reproduction |
|--------|----------|-----------------|
| Compute | 4× A100 80GB | Kaggle P100 16GB |
| Precision | bfloat16 (wide range) | float16 + GradScaler (overflow-prone) |
| Training | Full parameters via VERL/Ray | LoRA (r=16, alpha=32) via PyTorch |
| Batch size | 128 | 4 (with gradient accumulation) |
| Rollouts | 5 per prompt | 3-5 per prompt |
| Training samples | 5,120 per dataset | 512 per dataset |
| Learning rate | 5e-7 (full-param) | 2e-5 (LoRA) + cosine decay |
| Format threshold | 1.0 (multi-turn required) | 0.5 (single-turn unlocks F1) |
| N-ary extraction | GPT-4o-mini API | Local regex-based |
| Framework | VERL + Ray distributed | Custom single-GPU PyTorch |

## Challenges & Lessons Learned

Reproducing Graph-R1 on Kaggle's free tier (P100 16GB) over 20 iterations revealed:

1. **Silent evaluation bugs can mask all training progress**: The answer extraction bug (runs 8-19) meant the model was training correctly but evaluation always returned 0%. The fix was trivial (take last regex match instead of first), but the bug was invisible because the extracted `"..."` looked like a degenerate model output, not a template artifact.

2. **Float16 is not bfloat16**: The paper's A100s use bfloat16 (exponent range ±38), which handles the large values in log_softmax and policy ratios natively. Float16 (exponent range ±5) overflows at 65504, requiring explicit float32 upcasting and GradScaler — a P100-specific concern that no A100-trained codebase addresses.

3. **LoRA LR must be ~40x higher than full-param LR**: The paper's 5e-7 produced zero learning with LoRA. Standard LoRA LR of 2e-5 was needed.

4. **Exploration is the bottleneck, not optimization**: The model quickly learned the easy pattern (`<think>` + `<answer>`) but never discovered `<query>` usage for retrieval without SFT warmup. RL can only reinforce behaviors the model sometimes produces.

5. **SFT warmup is essential for format learning**: A brief supervised phase (30 steps) teaching the think → query → answer format provides the behavioral anchoring that RL then refines.

6. **GRPO advantage collapse is a real failure mode**: With few rollouts and low temperature, all rollouts converge to the same reward, zeroing out the gradient signal entirely. Temperature, rollout count, and normalization fallbacks all matter.

7. **Memory management requires architectural changes**: Per-rollout backward passes, gradient checkpointing, aggressive cache clearing, and sequence length capping are all necessary for 16GB VRAM — simply reducing batch size is not enough.

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
