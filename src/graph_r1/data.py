"""
Dataset loading and preprocessing for Graph-R1.

Handles the 6 FlashRAG benchmark datasets:
- 2WikiMultiHopQA, HotpotQA, Musique, NQ, PopQA, TriviaQA

Implements the preprocessing from script_process.py:
- Load raw train/dev/test splits
- Format into instruction-following format with <think>/<query>/<answer> tags
- Save as parquet for training
"""

import json
import os
from typing import Any

import pandas as pd
from tqdm import tqdm


DATASETS = [
    "2WikiMultiHopQA",
    "HotpotQA",
    "Musique",
    "NQ",
    "PopQA",
    "TriviaQA",
]

INSTRUCTION_TEMPLATE = """You are a helpful assistant. Answer the given question. You can query from knowledge base provided to you to answer the question.
You can query knowledge as many times as you want. You must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>. When you have the final answer, you can output the answer inside <answer>...</answer>. Question: {question}"""


def load_flashrag_dataset(data_dir: str, dataset_name: str, split: str = "train") -> list[dict]:
    """Load a FlashRAG dataset from JSON files.

    Expected file: {data_dir}/{dataset_name}/raw/{split}.json
    Each entry has 'question', 'golden_answers', and optionally 'gold_context'.
    """
    filepath = os.path.join(data_dir, dataset_name, "raw", f"{split}.json")
    if not os.path.exists(filepath):
        filepath_alt = os.path.join(data_dir, dataset_name, f"{split}.json")
        if os.path.exists(filepath_alt):
            filepath = filepath_alt
        else:
            raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    return data


def load_corpus(data_dir: str, dataset_name: str) -> list[dict]:
    """Load the corpus documents for a dataset.

    Expected file: {data_dir}/{dataset_name}/corpus.jsonl
    """
    filepath = os.path.join(data_dir, dataset_name, "corpus.jsonl")
    if not os.path.exists(filepath):
        return []

    docs = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def format_for_training(
    data: list[dict],
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Format raw QA data into the Graph-R1 training format.

    Each sample becomes:
    {
        "prompt": [{"role": "user", "content": instruction}],
        "ability": "multihop_qa",
        "reward_model": {"ground_truth": answers, "style": "rule"},
        "data_source": dataset_name,
        "extra_info": {"split": split, "index": i, "question": q, "answer": a}
    }
    """
    formatted = []
    samples = data[:max_samples] if max_samples else data

    for i, item in enumerate(tqdm(samples, desc=f"Formatting {dataset_name}/{split}")):
        question = item.get("question", "")
        answers = item.get("golden_answers", item.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        instruction = INSTRUCTION_TEMPLATE.format(question=question)

        formatted.append({
            "prompt": [{"role": "user", "content": instruction}],
            "ability": "multihop_qa",
            "reward_model": {
                "ground_truth": answers,
                "style": "rule",
            },
            "data_source": dataset_name,
            "extra_info": {
                "split": split,
                "index": i,
                "question": question,
                "answer": answers[0] if answers else "",
            },
        })

    return formatted


def save_as_parquet(data: list[dict], output_path: str) -> None:
    """Save formatted data as parquet file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for item in data:
        rows.append({
            "prompt": json.dumps(item["prompt"]),
            "ability": item["ability"],
            "reward_model": json.dumps(item["reward_model"]),
            "data_source": item["data_source"],
            "extra_info": json.dumps(item["extra_info"]),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")


def prepare_dataset(
    data_dir: str,
    dataset_name: str,
    output_dir: str,
    train_samples: int = 5120,
    test_samples: int = 128,
) -> dict[str, str]:
    """Full pipeline: load, format, and save a dataset."""
    output_paths = {}

    for split, max_n in [("train", train_samples), ("test", test_samples), ("dev", test_samples)]:
        try:
            data = load_flashrag_dataset(data_dir, dataset_name, split)
            formatted = format_for_training(data, dataset_name, split, max_samples=max_n)
            out_path = os.path.join(output_dir, dataset_name, "processed", f"{split}.parquet")
            save_as_parquet(formatted, out_path)
            output_paths[split] = out_path
        except FileNotFoundError:
            print(f"Skipping {dataset_name}/{split}: file not found")

    return output_paths


def load_parquet_dataset(path: str) -> list[dict]:
    """Load a processed parquet dataset back into dict format."""
    df = pd.read_parquet(path)
    data = []
    for _, row in df.iterrows():
        data.append({
            "prompt": json.loads(row["prompt"]),
            "ability": row["ability"],
            "reward_model": json.loads(row["reward_model"]),
            "data_source": row["data_source"],
            "extra_info": json.loads(row["extra_info"]),
        })
    return data
