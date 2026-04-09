"""
CodeNav GRPO Training Script.

Trains a language model to solve developer workflow tasks using
Group Relative Policy Optimization (GRPO) via TRL.

The model learns to:
  1. Read relevant files selectively
  2. Diagnose bugs before editing
  3. Make minimal targeted fixes
  4. Verify with tests before submitting

Usage:
    pip install trl transformers torch openenv-core

    # Train on easy tasks (recommended starting point)
    python train.py --task easy --model Qwen/Qwen2.5-0.5B-Instruct

    # Train on all tasks
    python train.py --task all --model Qwen/Qwen2.5-1.5B-Instruct

    # Resume from checkpoint
    python train.py --task easy --model Qwen/Qwen2.5-0.5B-Instruct --resume ./outputs/checkpoint-100

Environment variables:
    CODENAV_SPACE_URL   CodeNav HF Space URL (default: local environment)
    HF_TOKEN            HuggingFace token for pushing model checkpoints
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Imports — graceful error messages for missing deps
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    print("ERROR: trl not installed. Run: pip install trl>=0.12.0")
    sys.exit(1)

from server.codenav_environment import CodeNavEnvironment
from models import CodeNavAction, CodeNavObservation


# ---------------------------------------------------------------------------
# System prompt — same as inference.py for consistency
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert software developer working inside a Python codebase.
Your job is to find and fix bugs by following a careful workflow:

1. READ — Explore relevant files before acting. Never edit a file you haven't read.
2. DIAGNOSE — Submit a diagnosis explaining what is wrong and why before making any edit.
3. ACT — Make the minimal targeted fix. Change only what needs to change.
4. VERIFY — Run tests after your fix to confirm it works.
5. SUBMIT — Only submit when tests pass.

Respond with ONLY a valid JSON object — no explanation, no markdown, no code blocks.

Available actions:
  {"action_type": "read_file", "filename": "example.py"}
  {"action_type": "read_function", "filename": "example.py", "function_name": "my_func"}
  {"action_type": "search_codebase", "query": "search term"}
  {"action_type": "submit_diagnosis", "diagnosis": "The bug is X because Y"}
  {"action_type": "edit_code", "filename": "example.py", "old_code": "exact code", "new_code": "fix"}
  {"action_type": "run_tests"}
  {"action_type": "submit"}

CRITICAL: old_code must match file content EXACTLY. Read the file before editing.
Respond with ONLY the JSON object, nothing else.
"""


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[CodeNavAction]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        return CodeNavAction(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Episode runner — collects one full trajectory
# ---------------------------------------------------------------------------

def run_episode(
    model,
    tokenizer,
    task_id: str,
    scenario_index: Optional[int] = None,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run one full CodeNav episode with the given model.

    Returns:
        dict with:
            prompts     — list of prompt strings (one per step)
            completions — list of model completion strings
            rewards     — list of per-step rewards (float)
            final_score — float episode score
    """
    env = CodeNavEnvironment(task_id=task_id, scenario_index=scenario_index)
    reset_obs = env.reset()

    prompts = []
    completions = []
    rewards = []
    final_score = 0.0

    # Build opening prompt
    opening = (
        f"TASK: {reset_obs.message}\n\n"
        f"Available files: {reset_obs.available_files}\n"
        f"Max steps: {reset_obs.max_steps}\n\n"
        "Start by reading the relevant files. "
        "Respond with your first action as a JSON object."
    )

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": opening},
    ]

    steps_taken = 0

    while steps_taken < env._task["max_steps"]:
        # Format conversation for the model
        prompt_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate action
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse action
        action = parse_action(completion)
        if action is None:
            # Invalid JSON — small penalty, skip step
            prompts.append(prompt_text)
            completions.append(completion)
            rewards.append(-0.05)
            conversation.append({"role": "assistant", "content": completion})
            conversation.append({
                "role": "user",
                "content": "ERROR: Invalid JSON. Respond with ONLY a JSON object."
            })
            steps_taken += 1
            continue

        steps_taken += 1

        # Execute action
        obs = env.step(action)

        # Compute step reward
        step_reward = 0.0
        if obs.final_score is not None:
            step_reward = obs.final_score
            final_score = obs.final_score

        # Record trajectory
        prompts.append(prompt_text)
        completions.append(completion)
        rewards.append(step_reward)

        # Update conversation
        obs_text = _format_obs(obs, steps_taken)
        conversation.append({"role": "assistant", "content": completion})
        conversation.append({"role": "user", "content": obs_text})

        if obs.done or action.action_type == "submit":
            break

    return {
        "prompts": prompts,
        "completions": completions,
        "rewards": rewards,
        "final_score": final_score,
        "steps_taken": steps_taken,
    }


def _format_obs(obs: CodeNavObservation, step: int) -> str:
    """Format observation as text for the model."""
    parts = [f"[Step {step} result] {obs.message}"]
    parts.append(
        f"Status: step {obs.current_step}/{obs.max_steps} | "
        f"files_read={obs.files_read} | edits={obs.edits_made} | "
        f"diagnosis_submitted={obs.diagnosis_submitted} | tests_run={obs.tests_run}"
    )
    if obs.file_content:
        content = obs.file_content[:2000] + "\n...[truncated]" if len(obs.file_content) > 2000 else obs.file_content
        parts.append(f"File content:\n{content}")
    if obs.diagnosis_feedback:
        parts.append(f"Diagnosis feedback: {obs.diagnosis_feedback}")
    if obs.diff:
        parts.append(f"Diff:\n{obs.diff}")
    if obs.test_results:
        r = obs.test_results
        parts.append(f"Tests: {r['total_passed']} passed, {r['total_failed']} failed")
        if r.get("files"):
            for fname, fr in r["files"].items():
                for tname, result in (fr.get("tests") or {}).items():
                    if result["status"] != "PASS":
                        parts.append(f"  FAIL {tname}: {result.get('error','')}")
    if obs.final_score is not None:
        parts.append(f"\nFINAL SCORE: {obs.final_score}")
        parts.append("Episode complete.")
    else:
        parts.append("\nWhat is your next action? Respond with JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# GRPO Dataset — generates episodes on the fly
# ---------------------------------------------------------------------------

class CodeNavDataset(torch.utils.data.Dataset):
    """
    Generates CodeNav episode prompts for GRPO training.

    Each item is a dict with 'prompt' — the opening task description.
    The reward function runs the full episode and returns the score.
    """

    def __init__(self, task_id: str, size: int = 500, scenario_index: Optional[int] = None):
        self.task_id = task_id
        self.size = size
        self.scenario_index = scenario_index
        self._episodes = []
        self._build()

    def _build(self):
        """Pre-generate episode opening prompts."""
        from server.codenav_environment import CodeNavEnvironment
        for _ in range(self.size):
            env = CodeNavEnvironment(
                task_id=self.task_id,
                scenario_index=self.scenario_index
            )
            obs = env.reset()
            opening = (
                f"TASK: {obs.message}\n\n"
                f"Available files: {obs.available_files}\n"
                f"Max steps: {obs.max_steps}\n\n"
                "Start by reading the relevant files. "
                "Respond with your first action as a JSON object."
            )
            self._episodes.append({
                "prompt": opening,
                "task_id": self.task_id,
            })

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._episodes[idx]


# ---------------------------------------------------------------------------
# Reward function — runs a full episode and returns the score
# ---------------------------------------------------------------------------

def make_reward_fn(task_id: str, max_new_tokens: int = 256):
    """
    Returns a GRPO-compatible reward function.

    The reward function takes (prompts, completions) and returns
    a list of reward scores by running CodeNav episodes.
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Parse the action from the completion
            action = parse_action(completion)
            if action is None:
                rewards.append(-0.1)
                continue

            # Run a fresh episode starting with this action
            try:
                env = CodeNavEnvironment(task_id=task_id)
                env.reset()
                obs = env.step(action)

                if obs.final_score is not None:
                    rewards.append(obs.final_score)
                elif obs.success:
                    # Intermediate step — small positive for valid action
                    rewards.append(0.05)
                else:
                    rewards.append(-0.05)
            except Exception:
                rewards.append(-0.1)

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CodeNav GRPO Training")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model to train (default: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="easy",
        help="Which task to train on (default: easy)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./codenav-grpo-output",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total training steps (default: 200)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes per task in dataset (default: 500)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 CodeNav GRPO Training")
    print(f"   Model  : {args.model}")
    print(f"   Task   : {args.task}")
    print(f"   Device : {device}")
    print(f"   Steps  : {args.steps}")
    print(f"   Output : {args.output}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Build datasets
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print(f"\nBuilding episode datasets for tasks: {tasks}")
    datasets = []
    for task_id in tasks:
        ds = CodeNavDataset(task_id=task_id, size=args.episodes)
        datasets.append(ds)
        print(f"  {task_id}: {len(ds)} episodes")

    # Combine datasets if multiple tasks
    if len(datasets) == 1:
        train_dataset = datasets[0]
        primary_task = tasks[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(datasets)
        primary_task = "easy"  # default reward fn task for mixed training

    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        # GRPO-specific
        num_generations=4,        # completions per prompt to compare
        max_new_tokens=256,
        temperature=0.7,
        # Use our reward function
        reward_funcs=[make_reward_fn(primary_task)],
    )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    # Train
    print(f"\nStarting GRPO training for {args.steps} steps...")
    print("Watch the reward climb as the model learns the developer workflow.\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    final_path = os.path.join(args.output, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Model saved to: {final_path}")

    # Quick evaluation
    print("\nRunning quick evaluation on trained model...")
    model.eval()
    for task_id in tasks:
        result = run_episode(
            model=model,
            tokenizer=tokenizer,
            task_id=task_id,
            scenario_index=0,
            device=device,
        )
        print(f"  {task_id}: score={result['final_score']:.3f} steps={result['steps_taken']}")


if __name__ == "__main__":
    main()