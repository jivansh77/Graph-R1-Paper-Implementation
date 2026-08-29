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

### Runs 10-12: TPU Gradient Checkpointing Fixes (FAILED)
- **Date**: 2026-08-28
- **Error**: torch_xla 2.8.0 is incompatible with gradient checkpointing (multiple paths through `_get_device_module('xla')` fail)
- **Fixes tried**: use_reentrant=True, preserve_rng_state=False, and finally disabling gradient checkpointing entirely on TPU

### Run 13: 2WikiMultiHopQA - v13 (FAILED - Killed/OOM after 5.7h)
- **Date**: 2026-08-28
- **Runtime**: ~5.7 hours (20,587s)
- **Hardware**: Kaggle TPU v3-8 via PyTorch/XLA 2.8.0
- **Config**: 512 train samples, LR=2e-5, batch_size=4, mini_batch_size=2, num_rollouts=3
- **Hypergraph**: 15,792 entities, 9,747 hyperedges
- **SFT warmup**: Completed successfully in ~22 min (30 steps, loss 0.62→0.41)
- **Positive**: Model IS now using `<query>` tags after SFT warmup! Rollout output shows: `<think>I need to find...<query>Are Shokrab, Kermanshah...`
- **Root cause**: XLA autoregressive generation is extremely slow (~24 min per generation vs ~10-20s on CPU). XLA needs to trace/compile each unique graph shape, and autoregressive generation has changing tensor shapes at each token. Eventually killed by OOM after accumulating too much memory during the slow generation.
- **Fix for v14**: Move model to CPU for generation (rollouts + eval), keep only training forward/backward on TPU

### Run 14: 2WikiMultiHopQA - v14 (FAILED - TPU HBM OOM)
- **Date**: 2026-08-28
- **Runtime**: ~41 min (2469s)
- **Hardware**: Kaggle TPU v3-8 (15.75GB HBM per core)
- **Key change**: CPU generation working perfectly (rollouts in ~30-80s vs ~24 min on XLA)
- **SFT warmup**: Completed (30 steps, 31 min)
- **Progress**: Generated first GRPO rollouts successfully on CPU. Model using `<query>` tags and receiving knowledge.
- **Error**: `ValueError: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 17.14G of 15.75G hbm.`
- **Root cause**: Model (~3GB) + ref_model (~3GB) + activations/gradients = 17.14GB exceeds 15.75GB per TPU core
- **Fix for v15**: Keep ref_model on CPU (only needs no-grad forward pass), freeing ~3GB TPU HBM

### Run 15: 2WikiMultiHopQA - v15 (FAILED - TPU HBM OOM)
- **Date**: 2026-08-29
- **Runtime**: ~40 min (2406s)
- **Error**: `Used 16.58G of 15.75G hbm. Exceeded by 854.89M.`
- **Progress**: SFT warmup completed. Model using `<query>` tags, getting R=0.00 (up from -0.50).
- **Root cause**: XLA lazy execution accumulates computation graphs across all rollouts in compute_policy_loss without mark_step(). 6 rollouts × activations = massive peak memory.
- **Fix for v16**: Add `_sync_device()` after each rollout's backward pass to force XLA graph execution and memory release. Also cap max_seq to 2048 tokens.

### Run 16: 2WikiMultiHopQA - v16 (FAILED - Killed after ~4.9h)
- **Date**: 2026-08-29
- **Runtime**: ~4.9 hours (17,723s)
- **Hardware**: Kaggle TPU v3-8
- **Key changes**:
  - `_sync_device()` (xm.mark_step()) after each rollout backward in compute_policy_loss
  - Max sequence length capped at 2048 for policy loss computation
- **Progress**: No more OOM! Completed SFT warmup + 20 GRPO steps
  - SFT warmup: 30 steps in ~24 min, losses 0.62→0.26
  - Model using `<query>` tags and receiving knowledge
  - Rewards improving: -0.50 → -0.33 (step 10) → -0.25 (step 20)
  - Some rollouts achieving R=0.00 (format correct, getting knowledge)
- **Error**: `Killed` (process killed by system after 17,723s)
- **Root cause**: CPU generation too slow (~8-9 min per GRPO step). 256 steps would need ~35 hours, far exceeding the 9h TPU limit. Each step requires moving model CPU→generate→CPU→TPU for training.
- **Positive**: XLA mark_step fix worked — no more TPU HBM OOM. Training dynamics are healthy.

### Run 17: 2WikiMultiHopQA - v17 (FAILED - NaN crash)
- **Date**: 2026-08-29
- **Hardware**: Kaggle GPU P100 16GB
- **Runtime**: ~480s
- **Error**: `RuntimeError: probability tensor contains either inf, nan or element < 0` during generation
- **Root cause**: P100 uses float16 (max value 65504). After SFT warmup, model logits overflow float16 during sampling.
- **Fix**: Added `_SafeLogitsProcessor` to clamp logits to [-1e4, 1e4] during generation

### Run 18: 2WikiMultiHopQA - v18 (COMPLETED - 0% results)
- **Date**: 2026-08-29
- **Runtime**: ~7.1 hours (25,404s)
- **Hardware**: Kaggle GPU P100 16GB
- **Config**: 512 train samples, LR=2e-5, batch_size=4, mini_batch_size=2, num_rollouts=3
- **Training**: 116 GRPO steps completed, all with Loss=NaN, Reward=-1.0
- **Results**: EM=0.0%, F1=0.0%
- **Root cause analysis**: SafeLogitsProcessor fixed generation sampling crash, but `compute_policy_loss` operations (log_softmax, exp(ratio)) still computed in native float16, producing NaN. NaN gradients corrupted all model weights by step 10, causing all subsequent rollouts to produce nonsense.
- **SFT warmup worked correctly**: Step 0 rollouts showed model using `<query>` tags and achieving R=0.00 (format-correct). But first GRPO backward pass produced NaN, destroying the learned weights.

### Run 19: 2WikiMultiHopQA - v19 (PENDING)
- **Date**: 2026-08-29
- **Hardware**: Kaggle GPU P100 16GB
- **Key changes** (informed by analysis of official LHRLAB/Graph-R1 repository):
  1. **Cast all loss computation to float32**: `logits.float()` before log_softmax/exp in both compute_policy_loss and SFT warmup. Official code runs on A100 with bfloat16/FSDP which handles precision automatically; P100 needs explicit float32 upcasting.
  2. **GradScaler for float16**: Proper mixed-precision training with torch.cuda.amp.GradScaler for backward/optimizer steps.
  3. **Ratio clamping**: `torch.clamp(ratio, max=10.0)` to prevent extreme policy ratios.
- **Official repo findings** (LHRLAB/Graph-R1):
  - Uses VERL framework with FSDP (handles mixed precision automatically)
  - 4x A100 80GB with bfloat16 (wide dynamic range, no overflow)
  - LR 5e-7 for full-parameter, batch_size=128, 5 rollouts
  - Format reward: checks `<|im_start|>assistant\n...<|im_end|>` blocks, +0.5 per valid block
  - Answer reward: F1 score, only unlocked when format_reward == 1.0
  - Uses `kl_loss_type: low_var_kl` with coeff 0.001
