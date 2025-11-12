#!/usr/bin/env python3
"""
RAGEN WebArena Trainer — runs rollouts in WebArena and logs results.

Usage:
    python -m ragen.train_ragen_webarena --config configs/ragen_webarena.yaml --output_dir ./outputs
"""

import os
import sys
import json
import yaml
import argparse
import time
import random
import traceback
from pathlib import Path
from tqdm import tqdm

import torch

# Browser env API (WebArena)
from browser_env import ScriptBrowserEnv, create_id_based_action


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_config_dir_candidates():
    # Common locations when running via Modal or locally
    return [
        os.path.join(os.getcwd(), "webarena", "webarena", "config_files"),
        os.path.join(os.getcwd(), "webarena", "config_files"),
        "/root/work/webarena/webarena/config_files",
        "/root/work/webarena/config_files",
    ]


def resolve_config_path(cfg):
    # cfg may contain absolute paths already; prefer that
    cand = cfg.get("environment", {}).get("config_path")
    if cand and os.path.exists(cand):
        return cand

    # fallback: try candidate dirs
    for p in find_config_dir_candidates():
        if os.path.exists(p):
            return p

    raise FileNotFoundError("Could not find WebArena config_files directory. Checked candidates.")


def setup_env_from_cfg(cfg):
    env_cfg = cfg.get("environment", {})
    env = ScriptBrowserEnv(
        headless=env_cfg.get("headless", True),
        observation_type=env_cfg.get("observation_type", "accessibility_tree"),
        current_viewport_only=env_cfg.get("current_viewport_only", True),
        viewport_size={"width": 1280, "height": 720},
    )
    return env


def safe_device_and_dtype(prefer_cuda=True):
    use_cuda = torch.cuda.is_available() and prefer_cuda
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    return device, dtype


def load_llm(model_name: str, device: str, dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model {model_name} on {device} (dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Try to use torch_dtype for GPU; fallback gracefully
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    except Exception:
        print("⚠️ model.from_pretrained failed with torch_dtype; retrying without dtype...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def extract_action(action_text: str):
    """Simple robust extraction of an action phrase from model output.
    Returns an action string understood by WebArena (e.g., 'click [id]' or 'scroll [id]').
    """
    text = action_text.lower().strip()
    # Look for common verbs
    if "click" in text:
        return "click [id]"
    if "type" in text or "input" in text:
        return "type [id]"
    if "scroll" in text:
        return "scroll [id]"
    # fallback: choose click if nothing else
    return "click [id]"


def rollout_task(env, tokenizer, model, device, cfg, config_file, max_turns, sampling_cfg):
    """
    Run a single task (one config_file).
    Returns a dict with trajectory info and total reward.
    """
    obs, info = env.reset(options={"config_file": config_file})
    trajectory = []
    total_reward = 0.0

    for step in range(max_turns):
        # Observation text (shorten to reasonable length)
        text_obs = obs.get("text", "")
        prompt = text_obs[: sampling_cfg.get("prompt_max_chars", 2000)]

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=sampling_cfg.get("prompt_max_tokens", 1024))
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Sampling / generation
        gen_kwargs = dict(
            max_new_tokens=sampling_cfg.get("max_new_tokens", 32),
            do_sample=True,
            temperature=sampling_cfg.get("temperature", 0.7),
            top_p=sampling_cfg.get("top_p", 0.9),
            top_k=sampling_cfg.get("top_k", 0),
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        action_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        action = extract_action(action_text)

        # Create WebArena action and step
        act = create_id_based_action(action)
        obs_next, reward, terminated, truncated, info = env.step(act)

        trajectory.append(
            {
                "step": step,
                "action_text": action_text,
                "parsed_action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        total_reward += float(reward)
        obs = obs_next

        if terminated or truncated:
            break

    return {"trajectory": trajectory, "total_reward": total_reward}


def save_result(out_path: Path, record: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train RAGEN on WebArena (rollouts).")
    parser.add_argument("--config", type=str, default="configs/ragen_webarena.yaml")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    # Resolve config_files dir used by WebArena
    try:
        config_files_dir = resolve_config_path(cfg)
    except Exception as e:
        print("ERROR finding config_files:", e)
        raise

    print("Using WebArena config dir:", config_files_dir)

    # Setup environment
    env = setup_env_from_cfg(cfg)

    # Find all json config files to run
    configs = sorted([os.path.join(config_files_dir, f) for f in os.listdir(config_files_dir) if f.endswith(".json")])
    if not configs:
        raise RuntimeError(f"No JSON config files found under {config_files_dir}")

    # Model & device
    device, dtype = safe_device_and_dtype()
    device = torch.device(device)
    model_name = cfg.get("model", {}).get("name")
    if not model_name:
        raise ValueError("Model name not found in config under model.name")

    tokenizer, model = load_llm(model_name, device, dtype)

    # Sampling config
    sampling_cfg = cfg.get("sampling", {})
    sampling_cfg.setdefault("max_new_tokens", 32)
    sampling_cfg.setdefault("temperature", 0.7)
    sampling_cfg.setdefault("top_p", 0.9)
    sampling_cfg.setdefault("top_k", 0)
    sampling_cfg.setdefault("prompt_max_chars", 2000)
    sampling_cfg.setdefault("prompt_max_tokens", 1024)

    # Training loop over tasks (this script does rollouts / evaluation style)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "trajectories.jsonl"

    max_tasks = cfg.get("data", {}).get("train_size", len(configs))
    max_turns = cfg.get("environment", {}).get("max_turns", 10)

    # Optionally use only a subset (task_subset override from config)
    task_subset = cfg.get("environment", {}).get("task_subset")
    if task_subset:
        selected_cfgs = [configs[i] for i in task_subset if 0 <= i < len(configs)]
    else:
        selected_cfgs = configs[:max_tasks]

    print(f"Starting rollouts for {len(selected_cfgs)} tasks (max_turns={max_turns})")
    total_rewards = []

    for idx, config_file in enumerate(selected_cfgs):
        try:
            print(f"\n--- Task {idx} : {config_file}")
            result = rollout_task(env, tokenizer, model, device, cfg, config_file, max_turns, sampling_cfg)
            rec = {
                "task_idx": idx,
                "config_file": config_file,
                "total_reward": result["total_reward"],
                "num_steps": len(result["trajectory"]),
                "trajectory": result["trajectory"],
                "timestamp": time.time(),
            }
            save_result(results_path, rec)
            total_rewards.append(result["total_reward"])
            print(f"Task {idx} done. Reward={result['total_reward']:.2f} steps={len(result['trajectory'])}")
        except Exception:
            print("Exception during task rollout:")
            traceback.print_exc()
            # continue to next task after an error

    # Close environment
    try:
        env.close()
    except Exception:
        pass

    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    print(f"\nAll tasks finished. Avg reward: {avg_reward:.3f} over {len(total_rewards)} tasks.")
    # exit code 0
    return


if __name__ == "__main__":
    main()
