"""
Download and prepare FlashRAG datasets for Graph-R1 experiments.

Usage:
    python scripts/prepare_data.py --dataset 2WikiMultiHopQA --data_dir data/
    python scripts/prepare_data.py --dataset all --data_dir data/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph_r1.data import DATASETS, prepare_dataset


def download_from_huggingface(dataset_name: str, data_dir: str) -> None:
    """Download dataset from FlashRAG HuggingFace repo."""
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

    raw_dir = os.path.join(data_dir, dataset_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    try:
        ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", hf_key, trust_remote_code=True)

        for split_name in ds:
            output_path = os.path.join(raw_dir, f"{split_name}.json")
            if os.path.exists(output_path):
                print(f"  {split_name}.json already exists, skipping")
                continue

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
            print(f"  Saved {len(data)} examples to {split_name}.json")

    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print("  Trying alternative download...")
        download_alternative(dataset_name, data_dir)


def download_alternative(dataset_name: str, data_dir: str) -> None:
    """Alternative dataset download using direct URLs or kaggle."""
    raw_dir = os.path.join(data_dir, dataset_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    try:
        from datasets import load_dataset

        alt_names = {
            "2WikiMultiHopQA": ("Anon8281/2wiki-multihop-qa", None),
            "HotpotQA": ("hotpotqa/hotpot_qa", "distractor"),
            "NQ": ("google-research-datasets/natural_questions", "default"),
            "TriviaQA": ("mandarjoshi/trivia_qa", "rc"),
            "PopQA": (None, None),
            "Musique": (None, None),
        }

        hf_name, config = alt_names.get(dataset_name, (None, None))
        if hf_name:
            ds = load_dataset(hf_name, config, trust_remote_code=True)
            for split_name in ["train", "validation"]:
                if split_name not in ds:
                    continue
                out_name = "dev" if split_name == "validation" else split_name
                output_path = os.path.join(raw_dir, f"{out_name}.json")

                data = []
                for item in ds[split_name]:
                    q = item.get("question", "")
                    a = item.get("answer", item.get("golden_answers", ""))
                    if isinstance(a, str):
                        a = [a]
                    elif isinstance(a, dict):
                        a = a.get("aliases", [a.get("value", "")])
                    data.append({"question": q, "golden_answers": a})

                with open(output_path, "w") as f:
                    json.dump(data, f)
                print(f"  Saved {len(data)} examples to {out_name}.json")

    except Exception as e:
        print(f"  Alternative download also failed: {e}")
        print(f"  Please manually place {dataset_name} data in {raw_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Prepare FlashRAG datasets")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--train_samples", type=int, default=5120)
    parser.add_argument("--test_samples", type=int, default=128)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print(f"{'='*60}")

        print("Step 1: Downloading...")
        download_from_huggingface(name, args.data_dir)

        print("Step 2: Formatting for training...")
        try:
            prepare_dataset(
                args.data_dir, name, args.data_dir,
                train_samples=args.train_samples,
                test_samples=args.test_samples,
            )
        except Exception as e:
            print(f"  Formatting failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
