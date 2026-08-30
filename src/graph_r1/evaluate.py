"""
Evaluation module for Graph-R1.

Computes the four metrics from Section 5.1 / Appendix F:
- Exact Match (EM) - Eq. 31
- F1 Score - Eq. 32
- Retrieval Similarity (R-S) - Eq. 33
- Generation Evaluation (G-E) - Eq. 34 (7 dimensions via GPT-4o-mini)
"""

import json
import os
import re

import numpy as np

from .rewards import compute_em, compute_f1, normalize_answer


def extract_answer_from_output(output: str) -> str:
    """Extract the final answer from model output."""
    match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    return match.group(1).strip() if match else output.strip()


def extract_queries_from_output(output: str) -> list[str]:
    """Extract all retrieval queries from model output."""
    return re.findall(r"<query>(.*?)</query>", output, re.DOTALL)


def compute_em_metric(predictions: list[str], ground_truths: list[list[str]]) -> float:
    """EM: Exact Match (Eq. 31)."""
    scores = []
    for pred, gts in zip(predictions, ground_truths):
        score = max(compute_em(pred, gt) for gt in gts) if gts else 0.0
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def compute_f1_metric(predictions: list[str], ground_truths: list[list[str]]) -> float:
    """F1: Token-level F1 (Eq. 32)."""
    scores = []
    for pred, gts in zip(predictions, ground_truths):
        score = max(compute_f1(pred, gt) for gt in gts) if gts else 0.0
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def compute_retrieval_similarity(
    retrieved_knowledge: list[str],
    gold_knowledge: list[str],
    embedding_model=None,
) -> float:
    """R-S: Retrieval Similarity via cosine similarity of embeddings (Eq. 33).

    R-S = (1/N) * Σ cos(Enc(k_retr), Enc(k_gold))
    """
    if not retrieved_knowledge or not gold_knowledge or embedding_model is None:
        return 0.0

    ret_embs = embedding_model.encode(retrieved_knowledge)
    gold_embs = embedding_model.encode(gold_knowledge)

    ret_embs = ret_embs / np.linalg.norm(ret_embs, axis=1, keepdims=True)
    gold_embs = gold_embs / np.linalg.norm(gold_embs, axis=1, keepdims=True)

    similarities = []
    for r, g in zip(ret_embs, gold_embs):
        sim = np.dot(r, g)
        similarities.append(float(sim))

    return np.mean(similarities) if similarities else 0.0


def evaluate_dataset(
    predictions: list[str],
    ground_truths: list[list[str]],
    retrieved_knowledge: list[str] | None = None,
    gold_knowledge: list[str] | None = None,
    embedding_model=None,
) -> dict:
    """Run full evaluation suite on predictions."""
    results = {
        "em": compute_em_metric(predictions, ground_truths),
        "f1": compute_f1_metric(predictions, ground_truths),
        "num_samples": len(predictions),
    }

    if retrieved_knowledge and gold_knowledge and embedding_model:
        results["retrieval_similarity"] = compute_retrieval_similarity(
            retrieved_knowledge, gold_knowledge, embedding_model
        )

    per_sample = []
    for i, (pred, gts) in enumerate(zip(predictions, ground_truths)):
        sample_result = {
            "index": i,
            "prediction": pred,
            "ground_truths": gts,
            "em": max(compute_em(pred, gt) for gt in gts) if gts else 0.0,
            "f1": max(compute_f1(pred, gt) for gt in gts) if gts else 0.0,
        }
        per_sample.append(sample_result)

    results["per_sample"] = per_sample
    return results


def save_evaluation_results(results: dict, output_path: str) -> None:
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {output_path}")


def format_results_table(results: dict[str, dict], dataset_names: list[str] | None = None) -> str:
    """Format evaluation results as a readable table."""
    lines = []
    lines.append(f"{'Dataset':<20} {'EM':>8} {'F1':>8} {'R-S':>8} {'G-E':>8}")
    lines.append("-" * 56)

    datasets = dataset_names or sorted(results.keys())
    all_em, all_f1 = [], []

    for name in datasets:
        if name not in results:
            continue
        r = results[name]
        em = r.get("em", 0) * 100
        f1 = r.get("f1", 0) * 100
        rs = r.get("retrieval_similarity", 0) * 100
        ge = r.get("generation_eval", 0) * 100
        all_em.append(em)
        all_f1.append(f1)
        lines.append(f"{name:<20} {em:>7.1f}% {f1:>7.1f}% {rs:>7.1f}% {ge:>7.1f}%")

    if all_em:
        lines.append("-" * 56)
        lines.append(f"{'Average':<20} {np.mean(all_em):>7.1f}% {np.mean(all_f1):>7.1f}%")

    return "\n".join(lines)
