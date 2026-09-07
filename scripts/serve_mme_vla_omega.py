"""Serve a checkpoint from the isolated Omega-conditioned suite."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import tyro

from mme_vla_omega_suite.policies import policy_config as _policy_config
from mme_vla_omega_suite.training import config as _config
from mme_vla_suite.serving import websocket_policy_server


@dataclasses.dataclass
class Checkpoint:
    config: str = "mme_vla_omega_wa05_action_modulation"
    dir: str = ""


@dataclasses.dataclass
class Args:
    policy: Checkpoint
    port: int = 8000
    seed: int = 42
    default_prompt: str | None = None


def main(args: Args) -> None:
    if not args.policy.dir:
        raise ValueError("policy.dir is required")
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        Path(args.policy.dir),
        seed=args.seed,
        default_prompt=args.default_prompt,
    )
    logging.info("Serving Omega policy on port %d", args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
