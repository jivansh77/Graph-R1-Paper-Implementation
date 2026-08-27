"""
Graph-R1 Full Experiment Pipeline - Kaggle Notebook Script

This script is designed to run as a Kaggle notebook with GPU acceleration.
It implements the complete Graph-R1 pipeline:
1. Data preparation (download + format FlashRAG datasets)
2. Knowledge Hypergraph construction (chunking + extraction + embedding)
3. GRPO Training on Qwen2.5-1.5B-Instruct
4. Evaluation on all 6 benchmarks

Hardware: Kaggle GPU (T4 16GB or P100 16GB)
Expected runtime: ~4-6 hours for 1.5B model on 1 dataset
"""

# %% [markdown]
# # Graph-R1: Agentic GraphRAG via End-to-end Reinforcement Learning
# ## Reproduction Experiment

# %% Setup and Installation
import subprocess
import sys
import os

def install_packages():
    import torch as _t
    needs_torch_reinstall = False
    if _t.cuda.is_available():
        try:
            cap = _t.cuda.get_device_capability(0)
            if cap[0] < 7:
                needs_torch_reinstall = True
                print(f"GPU capability {cap} < 7.0, downgrading PyTorch for compatibility...")
        except Exception:
            pass

    if needs_torch_reinstall:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu121"],
                       capture_output=True)

    packages = [
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "faiss-cpu>=1.7.4",
        "FlagEmbedding>=1.2.0",
        "tiktoken>=0.5.0",
        "jsonlines>=4.0.0",
        "nltk>=3.8.0",
        "sentence-transformers>=2.5.0",
        "openai>=1.12.0",
        "matplotlib",
    ]
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                       capture_output=True)

install_packages()

# Clone our repo (use the feature branch with all code)
REPO_DIR = "/kaggle/working/Graph-R1-Paper-Implementation"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-b", "claude/graph-r1-reproduction-h5n4un",
                     "https://github.com/jivansh77/Graph-R1-Paper-Implementation.git",
                     REPO_DIR], capture_output=True)

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# %% Imports
import json
import time
import numpy as np
import torch
from datetime import datetime

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"GPU Memory: {total_mem / 1e9:.1f} GB")

from graph_r1.data import (
    DATASETS, load_flashrag_dataset, format_for_training,
    save_as_parquet, load_parquet_dataset
)
from graph_r1.hypergraph import KnowledgeHyperGraph
from graph_r1.retrieval import HypergraphRetriever
from graph_r1.rewards import compute_reward, compute_f1, compute_em, extract_answer
from graph_r1.grpo_trainer import GRPOConfig, GRPOTrainer
from graph_r1.evaluate import evaluate_dataset, format_results_table, save_evaluation_results

# %% Configuration
EXPERIMENT_CONFIG = {
    "dataset": "2WikiMultiHopQA",
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "train_samples": 1024,
    "test_samples": 128,
    "batch_size": 4,
    "mini_batch_size": 1,
    "num_rollouts": 3,
    "max_turns": 3,
    "num_epochs": 1,
    "learning_rate": 5e-7,
    "eval_steps": 20,
    "save_steps": 100,
    "use_lora": True,
    "lora_r": 16,
}

DATA_DIR = "/kaggle/working/data"
EXPERIMENT_DIR = "/kaggle/working/experiments"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

dataset_name = EXPERIMENT_CONFIG["dataset"]
print(f"\n{'='*60}")
print(f"Experiment: Graph-R1 on {dataset_name}")
print(f"Model: {EXPERIMENT_CONFIG['model_name']}")
print(f"Train samples: {EXPERIMENT_CONFIG['train_samples']}")
print(f"{'='*60}")

# %% [markdown]
# ## Step 1: Data Preparation

# %% Download and prepare dataset
print("\n--- Step 1: Data Preparation ---")

raw_dir = os.path.join(DATA_DIR, dataset_name, "raw")
os.makedirs(raw_dir, exist_ok=True)

try:
    from datasets import load_dataset

    hf_name_map = {
        "2WikiMultiHopQA": "2wikimultihopqa",
        "HotpotQA": "hotpotqa",
        "Musique": "musique",
        "NQ": "nq",
        "PopQA": "popqa",
        "TriviaQA": "triviaqa",
    }

    hf_key = hf_name_map.get(dataset_name, dataset_name.lower())
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", hf_key, trust_remote_code=True)

    for split_name in ds:
        output_path = os.path.join(raw_dir, f"{split_name}.json")
        if not os.path.exists(output_path):
            data = []
            for item in ds[split_name]:
                entry = {
                    "question": item.get("question", ""),
                    "golden_answers": item.get("golden_answers", item.get("answer", [])),
                }
                if "gold_context" in item:
                    entry["gold_context"] = item["gold_context"]
                data.append(entry)
            with open(output_path, "w") as f:
                json.dump(data, f)
            print(f"  Saved {len(data)} {split_name} examples")
        else:
            print(f"  {split_name}.json already exists")

    corpus_path = os.path.join(DATA_DIR, dataset_name, "corpus.jsonl")
    if not os.path.exists(corpus_path) and "corpus" in ds:
        with open(corpus_path, "w") as f:
            for item in ds["corpus"]:
                f.write(json.dumps(item) + "\n")

except Exception as e:
    print(f"Download error: {e}")
    print("Will use sample data for testing...")

# Format for training
print("Formatting data for training...")
train_data_raw = load_flashrag_dataset(DATA_DIR, dataset_name, "train")
test_data_raw = load_flashrag_dataset(DATA_DIR, dataset_name, "test")

train_data = format_for_training(
    train_data_raw, dataset_name, "train",
    max_samples=EXPERIMENT_CONFIG["train_samples"]
)
test_data = format_for_training(
    test_data_raw, dataset_name, "test",
    max_samples=EXPERIMENT_CONFIG["test_samples"]
)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# %% [markdown]
# ## Step 2: Knowledge Hypergraph Construction

# %% Build hypergraph
print("\n--- Step 2: Knowledge Hypergraph Construction ---")

hg_dir = os.path.join(EXPERIMENT_DIR, dataset_name)
hg = KnowledgeHyperGraph(working_dir=hg_dir)

# Collect documents from training data contexts
documents = []
for item in train_data_raw[:EXPERIMENT_CONFIG["train_samples"]]:
    ctx = item.get("gold_context", item.get("context", ""))
    if isinstance(ctx, list):
        for c in ctx:
            if isinstance(c, dict):
                text = c.get("content", c.get("text", str(c)))
            else:
                text = str(c)
            if text.strip() and len(text) > 50:
                documents.append(text)
    elif isinstance(ctx, str) and ctx.strip() and len(ctx) > 50:
        documents.append(ctx)

# Also try corpus
corpus_path = os.path.join(DATA_DIR, dataset_name, "corpus.jsonl")
if os.path.exists(corpus_path):
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                text = item.get("contents", item.get("content", item.get("text", "")))
                if text.strip() and len(text) > 50:
                    documents.append(text)

# Deduplicate
documents = list(set(documents))
print(f"Collected {len(documents)} unique documents")

# Chunk documents
print("Chunking documents...")
chunks = hg.chunk_documents(documents, chunk_size=1200, overlap=50)
print(f"Created {len(chunks)} chunks")

# Extract relations (using simple extraction for efficiency on Kaggle)
print("Extracting n-ary relations...")
import re

def simple_extraction_fn(prompt: str) -> str:
    """Lightweight extraction without API calls."""
    text = prompt.split("Text:")[-1] if "Text:" in prompt else prompt
    text = text.split("<|COMPLETE|>")[0]
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]

    output_parts = []
    entities_seen = set()

    for sent in sentences[:10]:
        output_parts.append(f'("hyperedge"<|>{sent})')

        words = sent.split()
        for j, word in enumerate(words):
            clean = word.strip(".,;:!?\"'()[]{}")
            if (clean and clean[0].isupper() and len(clean) > 2
                and clean.lower() not in {"the", "this", "that", "these", "those",
                                          "and", "but", "for", "not", "its", "his",
                                          "her", "was", "has", "had", "are"}
                and clean.lower() not in entities_seen):
                entities_seen.add(clean.lower())
                output_parts.append(
                    f'("entity"<|>{clean}<|>entity<|>{clean} referenced in the text)'
                )

    return "##".join(output_parts) + "<|COMPLETE|>"

t0 = time.time()
hg.extract_relations(chunks, simple_extraction_fn, batch_size=100)
extraction_time = time.time() - t0
print(f"Extraction done in {extraction_time:.1f}s: {len(hg.entities)} entities, {len(hg.hyperedges)} hyperedges")

# Build embeddings
print("Building embeddings with bge-large-en-v1.5...")
device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
hg.build_embeddings(device=device)
embed_time = time.time() - t0
print(f"Embeddings built in {embed_time:.1f}s")

# Save hypergraph
hg.save()
stats = hg.stats()
print(f"Hypergraph saved. Stats: {json.dumps(stats, indent=2)}")

# %% [markdown]
# ## Step 3: Set up Retrieval

# %% Initialize retriever
print("\n--- Step 3: Setting up Retriever ---")

from FlagEmbedding import FlagAutoModel

embedding_model = FlagAutoModel.from_finetuned(
    'BAAI/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    devices=device,
)

retriever = HypergraphRetriever(hg, embedding_model=embedding_model)

# Quick test
test_query = test_data[0]["extra_info"]["question"] if test_data else "Who directed the movie?"
test_results = retriever.retrieve(test_query, top_k=5)
print(f"Test retrieval for: '{test_query[:80]}...'")
print(f"  Retrieved {len(test_results)} facts")
for i, fact in enumerate(test_results[:3]):
    print(f"  [{i+1}] {fact['content'][:100]}...")

# %% [markdown]
# ## Step 4: GRPO Training

# %% Train with GRPO
print("\n--- Step 4: GRPO Training ---")

config = GRPOConfig(
    model_name=EXPERIMENT_CONFIG["model_name"],
    learning_rate=EXPERIMENT_CONFIG["learning_rate"],
    batch_size=EXPERIMENT_CONFIG["batch_size"],
    mini_batch_size=EXPERIMENT_CONFIG["mini_batch_size"],
    num_rollouts=EXPERIMENT_CONFIG["num_rollouts"],
    max_turns=EXPERIMENT_CONFIG["max_turns"],
    num_epochs=EXPERIMENT_CONFIG["num_epochs"],
    eval_steps=EXPERIMENT_CONFIG["eval_steps"],
    save_steps=EXPERIMENT_CONFIG["save_steps"],
    use_lora=EXPERIMENT_CONFIG["use_lora"],
    lora_r=EXPERIMENT_CONFIG["lora_r"],
    output_dir=CHECKPOINT_DIR,
)

trainer = GRPOTrainer(config=config, retriever=retriever)

print(f"Starting GRPO training...")
print(f"  Model: {config.model_name}")
print(f"  Batch size: {config.batch_size}")
print(f"  Rollouts per prompt: {config.num_rollouts}")
print(f"  Max turns: {config.max_turns}")
print(f"  Learning rate: {config.learning_rate}")

t0 = time.time()
training_metrics = trainer.train(train_data, eval_data=test_data)
training_time = time.time() - t0
print(f"\nTraining completed in {training_time:.1f}s ({training_time/3600:.1f}h)")

# %% [markdown]
# ## Step 5: Evaluation

# %% Evaluate
print("\n--- Step 5: Evaluation ---")

eval_metrics = trainer.evaluate(test_data)
print(f"\nFinal Evaluation Results on {dataset_name}:")
print(f"  EM: {eval_metrics['em']*100:.1f}%")
print(f"  F1: {eval_metrics['f1']*100:.1f}%")
print(f"  Samples: {eval_metrics['num_samples']}")

# Save results
results = {
    "dataset": dataset_name,
    "model": EXPERIMENT_CONFIG["model_name"],
    "metrics": eval_metrics,
    "config": EXPERIMENT_CONFIG,
    "training_time_seconds": training_time,
    "hypergraph_stats": stats,
    "timestamp": datetime.now().isoformat(),
}

results_path = os.path.join(EXPERIMENT_DIR, f"{dataset_name}_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved to {results_path}")

# %% Compare with paper results
print("\n--- Comparison with Paper (Table 2) ---")
paper_results_1_5b = {
    "2WikiMultiHopQA": {"em": 35.13, "f1": 65.73},
    "HotpotQA": {"em": 65.30, "f1": None},
    "Musique": {"em": 28.28, "f1": None},
    "NQ": {"em": None, "f1": 59.13},
    "PopQA": {"em": None, "f1": 66.46},
    "TriviaQA": {"em": None, "f1": 70.83},
}

if dataset_name in paper_results_1_5b:
    paper = paper_results_1_5b[dataset_name]
    print(f"\n{'Metric':<10} {'Ours':>10} {'Paper':>10} {'Delta':>10}")
    print("-" * 42)
    if paper.get("em") is not None:
        our_em = eval_metrics['em'] * 100
        print(f"{'EM':<10} {our_em:>9.1f}% {paper['em']:>9.1f}% {our_em - paper['em']:>+9.1f}%")
    if paper.get("f1") is not None:
        our_f1 = eval_metrics['f1'] * 100
        print(f"{'F1':<10} {our_f1:>9.1f}% {paper['f1']:>9.1f}% {our_f1 - paper['f1']:>+9.1f}%")

# %% Save training curves
print("\n--- Saving Training Curves ---")

if training_metrics:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [m["step"] for m in training_metrics]
    rewards = [m["mean_reward"] for m in training_metrics]
    losses = [m["loss"] for m in training_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, rewards, alpha=0.3, color='blue')
    window = min(10, len(rewards))
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title(f"Training Reward ({dataset_name})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, losses, alpha=0.3, color='red')
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(losses)), smoothed, color='red', linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Training Loss ({dataset_name})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(EXPERIMENT_DIR, f"{dataset_name}_training_curves.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved training curves to {fig_path}")

print("\n" + "="*60)
print("Experiment complete!")
print("="*60)
