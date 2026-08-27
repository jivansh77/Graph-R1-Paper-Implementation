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

(Results will be filled in as experiments complete)
