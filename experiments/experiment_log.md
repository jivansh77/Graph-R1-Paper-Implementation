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

### Run 7: 2WikiMultiHopQA - v7 (IN PROGRESS)
- **Date**: 2026-08-27
- **Fixes**:
  - Per-rollout backward (free graph after each rollout instead of accumulating)
  - Gradient checkpointing enabled
  - torch.cuda.empty_cache() after generation, training, and eval
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  - Reduced max_prompt_length=2048, max_response_length=1024
  - Reduced mid-training eval from 32 to 16 samples
