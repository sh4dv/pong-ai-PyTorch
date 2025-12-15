"""Utility to inspect saved DQN Pong models in a readable way.

Usage examples:
  python model_info.py                      # inspect default models/dqn_pong.pth
  python model_info.py --path models/checkpoints/checkpoint_ep100.pth
"""

import argparse
import os
import torch
from dqn_model import DQN
from config import INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE, MODEL_SAVE_PATH


def _fmt_size(bytes_count: int) -> str:
    if bytes_count <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(bytes_count)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def summarize_checkpoint(path: str) -> None:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")

    # Basic file info
    try:
        filesize = os.path.getsize(path)
        print(f"- File size: {_fmt_size(filesize)}")
    except OSError:
        pass

    # Known fields
    epsilon = checkpoint.get("epsilon")
    training_step = checkpoint.get("training_step")
    policy_sd = checkpoint.get("policy_net_state_dict")
    target_sd = checkpoint.get("target_net_state_dict")
    optim_sd = checkpoint.get("optimizer_state_dict")

    print("- Keys:", list(checkpoint.keys()))
    if epsilon is not None:
        print(f"- Epsilon: {epsilon}")
    if training_step is not None:
        print(f"- Training step: {training_step}")

    # Rebuild model to inspect shapes and parameter counts
    if policy_sd is None:
        print("No policy_net_state_dict in checkpoint; cannot summarize model weights.")
        return

    model = DQN(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
    missing, unexpected = model.load_state_dict(policy_sd, strict=False)
    if missing:
        print("- Missing keys when loading:", missing)
    if unexpected:
        print("- Unexpected keys when loading:", unexpected)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Parameters: {total_params} total | {trainable_params} trainable")

    print("- Layer shapes (policy_net):")
    for name, tensor in model.state_dict().items():
        print(f"    {name}: {tuple(tensor.shape)}")

    if optim_sd is not None:
        opt_class = optim_sd.get("param_groups", [{}])[0].get("params")
        print(f"- Optimizer state present: yes (param_groups={len(optim_sd.get('param_groups', []))})")
    else:
        print("- Optimizer state present: no")

    if target_sd is None:
        print("- target_net_state_dict missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved DQN Pong model")
    parser.add_argument("--path", type=str, default=MODEL_SAVE_PATH, help="Path to checkpoint (.pth)")
    args = parser.parse_args()

    summarize_checkpoint(args.path)


if __name__ == "__main__":
    main()
