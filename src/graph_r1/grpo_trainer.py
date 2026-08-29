"""
GRPO Training Pipeline for Graph-R1.

Implements Section 4.3: End-to-end RL Optimization using
Group Relative Policy Optimization (GRPO).

Key equations:
- GRPO objective J_GRPO(θ) (Eq. 11)
- Advantage Â(τ_i) = (R(τ_i) - mean) / std  (Eq. 11)
- Policy loss with clipped ratio (standard PPO loss)
- KL regularization toward reference policy

Training hyperparameters from Table 3 / Appendix G:
- Batch size: 128, LR: 5e-7 (full-param) / 2e-5 (LoRA), Rollout N: 5
- Max length: 4096, KL coeff (β): 0.001
- Clip range (ε): 0.2 (standard)
"""

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    learning_rate: float = 2e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    num_rollouts: int = 5
    max_prompt_length: int = 4096
    max_response_length: int = 4096
    max_turns: int = 5
    top_k_retrieval: int = 5
    clip_range: float = 0.2
    kl_coeff: float = 0.001
    num_epochs: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 10
    max_grad_norm: float = 1.0
    output_dir: str = "checkpoints"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32


class GRPOTrainer:
    """Implements GRPO training for Graph-R1.

    This is a simplified but faithful implementation suitable for
    single-GPU training on Kaggle. The full paper uses 4x A100 GPUs
    with VERL/Ray, but the core algorithm is identical.
    """

    def __init__(self, config: GRPOConfig, retriever=None):
        self.config = config
        self.retriever = retriever
        self.device, self.device_type = self._detect_device()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device_type == "tpu":
            dtype = torch.bfloat16
        elif self.device_type == "cuda":
            cap = torch.cuda.get_device_capability(0)
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        if config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if self.device_type != "tpu":
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

        ref_device = torch.device("cpu") if self.device_type == "tpu" else self.device
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(ref_device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_device = ref_device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = None

        self.global_step = 0
        self.training_log = []

    @staticmethod
    def _detect_device():
        """Detect the best available device: TPU > CUDA > CPU."""
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f"Using TPU device")
            return device, "tpu"
        except (ImportError, RuntimeError):
            pass
        if torch.cuda.is_available():
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda"), "cuda"
        print("Using CPU")
        return torch.device("cpu"), "cpu"

    def _sync_device(self):
        """Synchronize and clean up device state."""
        if self.device_type == "tpu":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        elif self.device_type == "cuda":
            torch.cuda.empty_cache()

    def _move_for_generation(self):
        """Move model to CPU for generation on TPU (XLA is extremely slow for autoregressive generation)."""
        if self.device_type == "tpu":
            self.model.to(torch.device("cpu"))
            import gc; gc.collect()
            return torch.device("cpu")
        return self.device

    def _move_for_training(self):
        """Move model back to training device after generation."""
        if self.device_type == "tpu":
            self.model.to(self.device)
            self._sync_device()

    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str], ground_truths: list) -> list[dict]:
        """Generate N rollout trajectories per prompt using the current policy.

        For each prompt, generates num_rollouts completions with multi-turn
        retrieval interaction. On TPU, generation runs on CPU (XLA is
        extremely slow for autoregressive decoding due to per-token recompilation).
        """
        from .agent import ToolEnv
        from .rewards import compute_reward

        all_rollouts = []
        self.model.eval()
        gen_device = self._move_for_generation()

        for prompt_idx, (prompt, gt) in enumerate(zip(prompts, ground_truths)):
            prompt_rollouts = []

            for rollout_idx in range(self.config.num_rollouts):
                env = ToolEnv(
                    retriever=self.retriever,
                    max_turns=self.config.max_turns,
                    top_k=self.config.top_k_retrieval,
                )

                current_text = prompt
                full_response = ""
                is_done = False

                for turn in range(self.config.max_turns):
                    if is_done:
                        break

                    inputs = self.tokenizer(
                        current_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.max_prompt_length,
                    ).to(gen_device)

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=min(512, self.config.max_response_length),
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    response = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=False,
                    )

                    eos_token = self.tokenizer.eos_token or "<|im_end|>"
                    if eos_token in response:
                        response = response[:response.index(eos_token)]

                    full_response += response
                    observation, is_done = env.step(response)

                    if not is_done and observation:
                        obs_text = f"\n<|im_start|>user\n<knowledge>{observation}</knowledge>\n<|im_end|>\n<|im_start|>assistant\n"
                        current_text = current_text + response + obs_text
                        full_response += obs_text

                    del inputs, outputs

                solution = prompt + full_response
                reward = compute_reward(solution, gt)

                prompt_rollouts.append({
                    "prompt": prompt,
                    "response": full_response,
                    "solution": solution,
                    "reward": reward,
                    "ground_truth": gt,
                    "prompt_idx": prompt_idx,
                })

                if self.global_step < 3 and rollout_idx == 0:
                    resp_preview = full_response[:200].replace('\n', ' ')
                    gt_str = gt[0] if isinstance(gt, list) else gt
                    print(f"    [Rollout] R={reward:.2f} | '{resp_preview}' | GT: '{gt_str[:60]}'")

            all_rollouts.extend(prompt_rollouts)

        self._move_for_training()
        self.model.train()
        self._sync_device()
        return all_rollouts

    def compute_advantages(self, rollouts: list[dict]) -> list[dict]:
        """Compute GRPO advantages: normalize rewards within each prompt group.

        Â(τ_i) = (R(τ_i) - mean({R(τ_j)})) / std({R(τ_j)})
        """
        groups = defaultdict(list)
        for r in rollouts:
            groups[r["prompt_idx"]].append(r)

        for prompt_idx, group in groups.items():
            rewards = [r["reward"] for r in group]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-6

            for r in group:
                r["advantage"] = (r["reward"] - mean_r) / std_r

        return rollouts

    def compute_policy_loss(self, rollouts: list[dict]) -> float:
        """Compute the GRPO clipped policy loss with KL regularization.

        Uses per-rollout backward to avoid accumulating computational graphs.
        """
        total_loss_scalar = 0.0
        num_valid = 0
        max_seq = self.config.max_prompt_length + self.config.max_response_length

        for rollout in rollouts:
            prompt_text = rollout["prompt"]
            response_text = rollout["response"]
            advantage = rollout["advantage"]

            full_text = prompt_text + response_text
            tok_kwargs = dict(return_tensors="pt", truncation=True, max_length=max_seq)
            prompt_encoding = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=self.config.max_prompt_length,
            )
            prompt_len = prompt_encoding["input_ids"].shape[1]

            encoding = self.tokenizer(full_text, **tok_kwargs)
            response_len = encoding["input_ids"].shape[1] - prompt_len

            if response_len <= 0:
                continue

            with torch.no_grad():
                ref_enc = {k: v.to(self.ref_device) for k, v in encoding.items()}
                ref_outputs = self.ref_model(**ref_enc)
                ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_response_tokens = ref_enc["input_ids"][:, prompt_len:]
                ref_token_log_probs = ref_log_probs.gather(
                    2, ref_response_tokens.unsqueeze(-1)
                ).squeeze(-1).to(self.device)
                del ref_outputs, ref_logits, ref_log_probs, ref_enc

            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.model(**encoding)
            logits = outputs.logits[:, prompt_len - 1:-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            response_tokens = encoding["input_ids"][:, prompt_len:]
            token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
            del outputs, logits, log_probs

            ratio = torch.exp(token_log_probs - ref_token_log_probs.detach())

            advantage_tensor = torch.tensor(advantage, device=self.device, dtype=torch.float32)
            pg_loss1 = -advantage_tensor * ratio
            pg_loss2 = -advantage_tensor * torch.clamp(
                ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            kl_div = (ref_token_log_probs.detach() - token_log_probs).mean()
            loss = pg_loss + self.config.kl_coeff * kl_div

            scaled_loss = loss / len(rollouts)
            scaled_loss.backward()
            total_loss_scalar += loss.item()
            num_valid += 1

            del encoding, token_log_probs, ref_token_log_probs, ratio, loss, scaled_loss

        if num_valid > 0:
            total_loss_scalar /= num_valid

        return total_loss_scalar

    def train_step(self, batch_prompts: list[str], batch_gts: list) -> dict:
        """Execute one GRPO training step.

        1. Generate N rollouts per prompt
        2. Compute rewards and advantages
        3. Update policy with clipped loss + KL (per-rollout backward)
        """
        rollouts = self.generate_rollouts(batch_prompts, batch_gts)
        self._sync_device()

        rollouts = self.compute_advantages(rollouts)

        self.model.train()
        self.optimizer.zero_grad()

        loss_value = self.compute_policy_loss(rollouts)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self._sync_device()

        rewards = [r["reward"] for r in rollouts]
        advantages = [r["advantage"] for r in rollouts]
        current_lr = self.optimizer.param_groups[0]["lr"]

        metrics = {
            "loss": loss_value,
            "mean_reward": np.mean(rewards),
            "max_reward": max(rewards),
            "std_reward": np.std(rewards),
            "mean_advantage": np.mean(advantages),
            "num_rollouts": len(rollouts),
            "lr": current_lr,
            "step": self.global_step,
        }

        self.global_step += 1
        self.training_log.append(metrics)

        return metrics

    def sft_warmup(self, train_data: list[dict], num_steps: int = 20):
        """Supervised fine-tuning warmup to teach the model the expected output format.

        Creates synthetic examples showing the think→query→answer pattern
        and trains the model to produce them via next-token prediction.
        """
        print(f"\n--- SFT Warmup ({num_steps} steps) ---")
        self.model.train()

        for step in range(num_steps):
            self.optimizer.zero_grad()
            item = train_data[step % len(train_data)]
            if isinstance(item["prompt"], list):
                prompt_text = item["prompt"][0]["content"]
            else:
                prompt_text = item["prompt"]
            gt = item["reward_model"]["ground_truth"]
            gt_str = gt[0] if isinstance(gt, list) and gt else str(gt)
            question = item.get("extra_info", {}).get("question", "the question")

            prompt = f"<|im_start|>user\n{prompt_text}\n<|im_end|>\n<|im_start|>assistant\n"
            if step % 2 == 0:
                target_response = (
                    f"<think>I need to find information to answer: {question}</think>\n"
                    f"<query>{question}</query>"
                )
            else:
                target_response = (
                    f"<think>Based on the information I have, the answer is {gt_str}.</think>\n"
                    f"<answer>{gt_str}</answer>"
                )
            full_text = prompt + target_response + self.tokenizer.eos_token

            encoding = self.tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=self.config.max_prompt_length + 256,
            ).to(self.device)

            prompt_enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.config.max_prompt_length,
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            outputs = self.model(**encoding)
            logits = outputs.logits[:, prompt_len - 1:-1, :]
            targets = encoding["input_ids"][:, prompt_len:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self._sync_device()

            if step % 5 == 0:
                print(f"  SFT step {step}: loss={loss.item():.4f}")

            del encoding, outputs, logits, loss

        self._sync_device()
        print("SFT warmup complete.\n")

    def train(self, train_data: list[dict], eval_data: list[dict] | None = None,
              max_train_hours: float = 7.0) -> list[dict]:
        """Full training loop over the dataset."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        num_batches = len(train_data) // self.config.batch_size
        mini_batches_per_batch = max(1, self.config.batch_size // max(1, self.config.mini_batch_size))
        total_steps = num_batches * mini_batches_per_batch * self.config.num_epochs
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps), eta_min=1e-6)
        print(f"Training plan: {total_steps} steps, LR {self.config.learning_rate} → 1e-6 (cosine)")

        all_metrics = []
        train_start = time.time()

        for epoch in range(self.config.num_epochs):
            indices = np.random.permutation(len(train_data))

            for batch_idx in range(num_batches):
                elapsed_hours = (time.time() - train_start) / 3600
                if elapsed_hours >= max_train_hours:
                    print(f"Training time limit ({max_train_hours}h) reached at step {self.global_step}. Stopping to allow evaluation.")
                    self.save_checkpoint()
                    self.save_training_log()
                    return all_metrics

                start = batch_idx * self.config.batch_size
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch = [train_data[i] for i in batch_indices]
                prompts = []
                gts = []
                for item in batch:
                    if isinstance(item["prompt"], list):
                        prompt_text = item["prompt"][0]["content"]
                    else:
                        prompt_text = item["prompt"]
                    prompts.append(f"<|im_start|>user\n{prompt_text}\n<|im_end|>\n<|im_start|>assistant\n")
                    gts.append(item["reward_model"]["ground_truth"])

                mini_batch_size = min(self.config.mini_batch_size, len(prompts))
                for mb_start in range(0, len(prompts), mini_batch_size):
                    mb_end = min(mb_start + mini_batch_size, len(prompts))
                    mb_prompts = prompts[mb_start:mb_end]
                    mb_gts = gts[mb_start:mb_end]

                    metrics = self.train_step(mb_prompts, mb_gts)
                    all_metrics.append(metrics)

                    if self.global_step % 10 == 0:
                        print(
                            f"Step {self.global_step} | "
                            f"Loss: {metrics['loss']:.6f} | "
                            f"Reward: {metrics['mean_reward']:.4f} (max {metrics['max_reward']:.4f}) | "
                            f"LR: {metrics['lr']:.2e}"
                        )

                if self.config.eval_steps and self.global_step % self.config.eval_steps == 0:
                    if eval_data:
                        eval_metrics = self.evaluate(eval_data[:16])
                        print(f"  Eval F1: {eval_metrics.get('f1', 0):.4f}")
                        self._sync_device()

                if self.config.save_steps and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

        self.save_checkpoint()
        self.save_training_log()
        return all_metrics

    @torch.no_grad()
    def evaluate(self, eval_data: list[dict]) -> dict:
        """Evaluate the current model on a set of examples."""
        from .rewards import compute_f1, compute_em, extract_answer

        self.model.eval()
        gen_device = self._move_for_generation()
        predictions = []
        references = []

        for item in tqdm(eval_data, desc="Evaluating"):
            if isinstance(item["prompt"], list):
                prompt_text = item["prompt"][0]["content"]
            else:
                prompt_text = item["prompt"]
            prompt = f"<|im_start|>user\n{prompt_text}\n<|im_end|>\n<|im_start|>assistant\n"
            gt = item["reward_model"]["ground_truth"]

            current_text = prompt
            for turn in range(self.config.max_turns):
                inputs = self.tokenizer(
                    current_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                ).to(gen_device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=False,
                )
                del inputs, outputs

                eos_token = self.tokenizer.eos_token or "<|im_end|>"
                if eos_token in response:
                    response = response[:response.index(eos_token)]

                if re.search(r"<answer>.*?</answer>", response, re.DOTALL):
                    current_text += response
                    break

                query_match = re.search(r"<query>(.*?)</query>", response, re.DOTALL)
                if query_match and self.retriever:
                    query = query_match.group(1).strip()
                    facts = self.retriever.retrieve(query, top_k=self.config.top_k_retrieval)
                    knowledge = self.retriever.format_knowledge(facts)
                    obs = f"\n<|im_start|>user\n<knowledge>{knowledge}</knowledge>\n<|im_end|>\n<|im_start|>assistant\n"
                    current_text += response + obs
                else:
                    current_text += response
                    break

            answer = extract_answer(current_text) or ""
            predictions.append(answer)
            references.append(gt)

            if len(predictions) <= 3:
                gt_str = gt[0] if isinstance(gt, list) else gt
                print(f"  [Eval sample {len(predictions)}] Pred: '{answer[:80]}' | Gold: '{gt_str[:80]}'")

        self._move_for_training()

        em_scores = []
        f1_scores = []
        for pred, gt_list in zip(predictions, references):
            if isinstance(gt_list, str):
                gt_list = [gt_list]
            em_scores.append(max(compute_em(pred, gt) for gt in gt_list) if gt_list else 0.0)
            f1_scores.append(max(compute_f1(pred, gt) for gt in gt_list) if gt_list else 0.0)

        self.model.train()
        return {
            "em": np.mean(em_scores),
            "f1": np.mean(f1_scores),
            "num_samples": len(predictions),
        }

    def save_checkpoint(self, path: str | None = None):
        """Save model checkpoint."""
        save_path = path or os.path.join(self.config.output_dir, f"step_{self.global_step}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=2)
        print(f"Saved checkpoint to {save_path}")

    def save_training_log(self):
        """Save training metrics log."""
        log_path = os.path.join(self.config.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
