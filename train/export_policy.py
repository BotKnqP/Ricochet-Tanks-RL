"""Export Stable-Baselines3 PPO policies for the browser watch mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch.nn as nn
from stable_baselines3 import PPO


def _linear_layer(layer: nn.Linear, act: str) -> dict[str, Any]:
    weight = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    return {
        "w": weight.astype(float).tolist(),
        "b": bias.astype(float).tolist(),
        "act": act,
    }


def extract_policy(model: PPO) -> dict[str, Any]:
    """Extract the policy-network logits path used by SB3's MlpPolicy."""
    layers: list[dict[str, Any]] = []
    for module in model.policy.mlp_extractor.policy_net:
        if isinstance(module, nn.Linear):
            layers.append(_linear_layer(module, "tanh"))
        elif isinstance(module, nn.Tanh):
            continue
        else:
            raise TypeError(f"unsupported policy_net module for browser export: {module!r}")
    layers.append(_linear_layer(model.policy.action_net, "linear"))
    return {"layers": layers}


def write_policy_files(policy: dict[str, Any], run_dir: str | Path, status: dict[str, Any] | None = None) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    policy_json = json.dumps(policy, separators=(",", ":"))
    (run_path / "live_policy.json").write_text(policy_json, encoding="utf-8")
    (run_path / "live_policy.js").write_text(
        "window.RICOCHET_POLICIES = window.RICOCHET_POLICIES || {};\n"
        f"window.RICOCHET_POLICIES.live = {policy_json};\n",
        encoding="utf-8",
    )

    if status is not None:
        status_json = json.dumps(status, separators=(",", ":"))
        (run_path / "live_status.json").write_text(status_json, encoding="utf-8")
        (run_path / "live_status.js").write_text(
            f"window.RICOCHET_LIVE_STATUS = {status_json};\n",
            encoding="utf-8",
        )


def export_model(model: PPO, run_dir: str | Path, status: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = extract_policy(model)
    write_policy_files(policy, run_dir, status)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ppo_level1_open_stationary"))
    args = parser.parse_args()

    model = PPO.load(args.model)
    export_model(model, args.run_dir, {
        "timesteps": int(getattr(model, "num_timesteps", 0)),
        "phase": "level1_open_stationary",
        "opponent": "stationary",
        "arenaMode": "open",
        "ep_rew_mean": None,
        "ep_len_mean": None,
    })


if __name__ == "__main__":
    main()
