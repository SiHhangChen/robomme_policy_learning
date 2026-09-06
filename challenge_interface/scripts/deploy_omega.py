"""Serve the three-view Omega policy through the challenge interface.

Example:
``uv run python -m challenge_interface.scripts.deploy_omega --checkpoint-dir runs/omega/79999``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from challenge_interface.omega_policy import MyOmegaPolicy_for_CVPR_Challenge
from challenge_interface.server import PolicyServer
from challenge_interface.server_http import PolicyHTTPServer
from mme_vla_omega_suite.policies.policy_config import create_trained_policy
from mme_vla_omega_suite.training.config import get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the three-view Omega policy for the CVPR challenge")
    parser.add_argument("--transport", choices=("websocket", "http"), default="websocket")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", default="mme_vla_omega_action_modulation")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config not in {
        "mme_vla_omega_action_modulation",
        "mme_vla_omega_ts01_action_modulation",
        "mme_vla_omega_wa01_action_modulation",
        "mme_vla_omega_wa01_action_condition",
    }:
        raise ValueError(f"Unsupported Omega challenge config: {args.config}")
    model = create_trained_policy(get_config(args.config), args.checkpoint_dir, seed=args.seed)
    policy = MyOmegaPolicy_for_CVPR_Challenge(model=model)
    server_cls = PolicyHTTPServer if args.transport == "http" else PolicyServer
    server_cls(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
