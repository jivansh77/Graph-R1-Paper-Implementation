"""
Reward computation for Graph-R1 GRPO training.

Implements Section 4.3 of the paper:
- Format Reward R_format(τ) (Eq. 12): checks think/query/answer tag structure
- Answer Reward R_answer(a_T^ans): token-level F1 score
- Overall Reward R(τ) = -1.0 + R_format(τ) + 𝟙{R_format=1} · R_answer (Eq. 13)
"""

import re
import string
from collections import Counter


def normalize_answer(answer: str) -> str:
    """Normalize answer text for comparison: lowercase, remove articles/punctuation."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def extract_answer(text: str) -> str:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_blocks(solution: str) -> list[str]:
    """Extract assistant message blocks from the full solution string."""
    blocks = []
    parts = re.split(r"<\|im_start\|>assistant", solution)
    for part in parts[1:]:
        end_idx = part.find("<|im_end|>")
        if end_idx != -1:
            blocks.append(part[:end_idx].strip())
        else:
            blocks.append(part.strip())
    if not blocks:
        blocks = [solution]
    return blocks


def compute_format_reward(solution: str) -> float:
    """Compute format reward R_format(τ) per Eq. 12.

    Awards 0.5 per valid step that includes proper tag structure:
    - Intermediate steps: <think>...</think> followed by <query>...</query>
    - Final step: <think>...</think> followed by <answer>...</answer>
    Capped at 1.0.
    """
    blocks = extract_blocks(solution)
    if not blocks:
        return 0.0

    total_score = 0.0
    num_blocks = len(blocks)

    for i, block in enumerate(blocks):
        is_last = (i == num_blocks - 1)
        has_think = bool(re.search(r"<think>.*?</think>", block, re.DOTALL))

        if is_last:
            has_answer = bool(re.search(r"<answer>.*?</answer>", block, re.DOTALL))
            if has_think and has_answer:
                total_score += 0.5
        else:
            has_query = bool(re.search(r"<query>.*?</query>", block, re.DOTALL))
            if has_think and has_query:
                total_score += 0.5

    return min(1.0, total_score)


def compute_answer_reward(solution: str, ground_truth: str | list[str]) -> float:
    """Compute answer reward R_answer using F1 score.

    If ground_truth is a list, takes the max F1 across all references.
    Penalizes degenerate answers (ellipsis, single punctuation, etc.).
    """
    predicted = extract_answer(solution)
    if not predicted:
        return 0.0

    normalized = predicted.strip().strip(".")
    if not normalized or len(normalized) <= 1:
        return -0.25

    if isinstance(ground_truth, list):
        return max(compute_f1(predicted, gt) for gt in ground_truth) if ground_truth else 0.0
    return compute_f1(predicted, ground_truth)


def compute_reward(solution: str, ground_truth: str | list[str],
                    format_threshold: float = 0.5) -> float:
    """Compute overall reward R(τ) per Eq. 13.

    R(τ) = -1.0 + R_format(τ) + 𝟙{R_format(τ) >= threshold} · R_answer(a_T^ans)

    The paper uses threshold=1.0 (requiring multi-turn format). For
    compute-constrained settings (LoRA on single GPU), threshold=0.5
    allows single-turn <think>+<answer> to unlock answer credit, giving
    the model a gradient signal to learn answer quality earlier.
    """
    r_format = compute_format_reward(solution)
    r_answer = compute_answer_reward(solution, ground_truth)
    indicator = 1.0 if r_format >= format_threshold else 0.0
    return -1.0 + r_format + indicator * r_answer


def compute_reward_for_batch(
    solutions: list[str],
    ground_truths: list[str | list[str]],
    format_threshold: float = 0.5,
) -> list[float]:
    """Compute rewards for a batch of solutions."""
    return [compute_reward(sol, gt, format_threshold) for sol, gt in zip(solutions, ground_truths)]


def compute_em_score(predictions: list[str], ground_truths: list[list[str]]) -> float:
    """Compute average exact match across a dataset."""
    scores = []
    for pred, gts in zip(predictions, ground_truths):
        score = max(compute_em(pred, gt) for gt in gts) if gts else 0.0
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_f1_score(predictions: list[str], ground_truths: list[list[str]]) -> float:
    """Compute average F1 across a dataset."""
    scores = []
    for pred, gts in zip(predictions, ground_truths):
        score = max(compute_f1(pred, gt) for gt in gts) if gts else 0.0
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0
