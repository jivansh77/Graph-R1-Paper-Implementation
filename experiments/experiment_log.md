# Graph-R1 Experiment Log

## Experiment Plan

### Phase 1: Data Preparation & Hypergraph Construction
- Download 6 FlashRAG datasets (5120 train / 128 test each)
- Build Knowledge Hypergraph for each dataset
  - Document chunking (1200 tokens, 50 overlap)
  - N-ary relation extraction
  - BGE-large-en-v1.5 embeddings + FAISS indices

### Phase 2: GRPO Training
- Base model: Qwen2.5-1.5B-Instruct (primary), Qwen2.5-3B-Instruct (if compute allows)
- Training config: batch_size=128 (paper) / adapted for single GPU
- LoRA fine-tuning (r=16, alpha=32) for memory efficiency
- 1 epoch, GRPO with 5 rollouts per prompt
- Reward: R(τ) = -1.0 + R_format + I{R_format=1} * R_answer(F1)

### Phase 3: Evaluation
- Metrics: EM, F1, R-S, G-E
- Compare with Table 2 from paper

## Paper Results (Table 2) - Qwen2.5-1.5B-Instruct

| Dataset | EM | F1 | G-E |
|---------|----|----|-----|
| 2WikiMultiHopQA | 35.13 | 65.73 | 64.38 |
| HotpotQA | 65.30 | - | - |
| Musique | 28.28 | - | - |
| NQ | - | 59.13 | - |
| PopQA | - | 66.46 | - |
| TriviaQA | - | 70.83 | - |
| **Average** | **31.90** | **40.09** | - |

## Differences from Original Implementation
1. **Compute**: Kaggle single GPU (T4/P100 16GB) vs. 4x A100 80GB
2. **Training**: LoRA fine-tuning vs. full parameter GRPO
3. **Batch size**: Reduced from 128 to 4-16 (with gradient accumulation)
4. **Rollouts**: Reduced from 5 to 3 for memory
5. **Extraction**: Simplified local extraction vs. GPT-4o-mini API calls
6. **Framework**: Custom PyTorch implementation vs. VERL/Ray distributed

## Experiment Runs

### Run 1: 2WikiMultiHopQA - v1 (FAILED)
- **Date**: 2026-08-27
- **Kernel**: jivanshc/graph-r1-reproduction-2wikimultihopqa v1
- **Error**: PyTorch 2.10.0+cu128 incompatible with P100 (sm_60, requires >=sm_70)
- **Root cause**: Kaggle's default PyTorch requires CUDA capability >= 7.0, P100 has 6.0
- **Additional error**: `total_mem` attribute renamed to `total_memory` in newer PyTorch

### Run 2: 2WikiMultiHopQA - v2 (FAILED)
- **Date**: 2026-08-27
- **Error**: PyTorch module caching, no test split in FlashRAG
- **Fixes**: Moved GPU check before torch import, dev split fallback

### Run 3: 2WikiMultiHopQA - v3 (FAILED)
- **Error**: `RuntimeError: operator torchvision::nms does not exist`
- **Fix**: Install torchvision==0.19.0 + torchaudio==2.4.0 matching torch 2.4.0

### Run 4: 2WikiMultiHopQA - v4 (FAILED)
- **Error**: `TypeError: BertModel.__init__() got an unexpected keyword argument 'dtype'`
- **Fix**: Removed transformers version pin

### Run 5: 2WikiMultiHopQA - v5 (FAILED)
- **Error**: `ImportError: incompatible torchao version`
- **Fix**: Uninstall torchao after torch downgrade

### Run 6: 2WikiMultiHopQA - v6 (FAILED - OOM)
- **Date**: 2026-08-27
- **Runtime**: ~93 minutes (5600s)
- **Progress**: Successfully completed data prep, hypergraph (25163 entities, 18544 hyperedges), embeddings, model loading, and 80+ GRPO training steps
- **Training observations**:
  - Rewards improving: -1.0 → -0.8333 (step 40) → -0.6667 (step 60)
  - Eval F1 remained 0.0 (model hasn't learned format yet)
  - Each step ~40s, eval ~7min
- **Error**: `torch.OutOfMemoryError: CUDA out of memory` at step ~85
- **Root cause**: compute_policy_loss accumulated all rollout loss tensors keeping full computational graphs in memory; no gradient checkpointing; no cache clearing

### Run 7: 2WikiMultiHopQA - v7 (SKIPPED)
- **Date**: 2026-08-27
- **Note**: Superseded by v8 with reduced training samples

### Run 8: 2WikiMultiHopQA - v8 (COMPLETED - 0% results)
- **Date**: 2026-08-27
- **Runtime**: ~3.4 hours
- **Config**: 256 train samples, batch_size=4, mini_batch_size=2, num_rollouts=3, LR=5e-7
- **Hypergraph**: 9,525 entities, 5,212 hyperedges
- **Training**: 128 steps completed
- **Results**: EM=0.0%, F1=0.0%
- **Diagnosis**: Three root causes for zero performance:
  1. **LR too low for LoRA**: 5e-7 (paper's full-param rate) produced losses of ~1e-7, effectively zero parameter updates. LoRA needs ~2e-5.
  2. **Format reward too sparse**: Required multi-turn format (R_format=1.0) to unlock answer reward. Single-turn think+answer only gets 0.5, so answer F1 never factors in.
  3. **Advantage signal killed**: With most rollouts getting -1.0 and group normalization, advantages were ~1e-16 (floating point noise).
- **Training dynamics**: Mean reward oscillated -1.0 to -0.75. Model occasionally produced `<think>`+`<answer>` (reward=-0.5) but gradient signal too weak to reinforce.

### Run 9: 2WikiMultiHopQA - v9 (COMPLETED - 0% results)
- **Date**: 2026-08-28
- **Runtime**: 5.4 hours (19,359s)
- **Config**: 512 train samples, LR=2e-5, format_threshold=0.5, cosine scheduler
- **Hypergraph**: 15,792 entities, 9,747 hyperedges
- **Training**: 256 steps completed
- **Results**: EM=0.0%, F1=0.0%
- **Progress vs v8**:
  - Losses now meaningful (1e-3 to 1e-2 range vs v8's ~1e-7) — LR fix worked
  - Model learned `<think>`+`<answer>` format consistently (max_reward always -0.5)
  - Mean reward improved to -0.58 to -0.75 range (vs -0.83 to -1.0 in v8)
- **Root cause**: Model produces `<answer>` tags but content has 0% F1. Never learned to use `<query>` tags for retrieval. Without retrieval, the 1.5B model can't answer multi-hop questions from its own knowledge. Local optimum: format-correct, content-wrong.
- **Diagnosis**: The instruction template describes the format but doesn't show an example. The model learns the easy part (think+answer) but never discovers the harder pattern (think+query→retrieve→think+answer).

### Run 10: 2WikiMultiHopQA - v10 (PENDING - switched to TPU)
- **Date**: 2026-08-28
- **Status**: Pushed to Kaggle with TPU accelerator (GPU quota exhausted at 30h/week)
- **Hardware**: Kaggle TPU v3-8 (8 cores, 128GB HBM total) via PyTorch/XLA
- **Fixes**:
  - Few-shot prompt: Concrete example in instruction template showing the full think→query→knowledge→think→answer pattern
  - SFT warmup: 30 steps of supervised fine-tuning before RL, alternating between query-generation and answer-generation targets
  - Debug output: First 3 rollout responses and eval predictions logged with ground truth
  - TPU support: Device-agnostic trainer with automatic TPU/GPU/CPU detection, bfloat16 on TPU, XLA mark_step synchronization
  - Embeddings run on CPU (FAISS and FlagEmbedding don't support XLA), model training on TPU
- **Expected impact**: SFT warmup should teach the model to produce `<query>` tags, enabling retrieval during RL rollouts and giving the model access to knowledge needed for correct answers. TPU bfloat16 may also help with training stability compared to GPU float16.
