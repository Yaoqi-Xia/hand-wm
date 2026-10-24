"""Reading generated clips and replaying them in MuJoCo (the answer key)."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def list_shards(data_dir) -> list[Path]:
    return sorted(Path(data_dir).glob("shard_[0-9][0-9][0-9][0-9][0-9].npz"))


def load_shard(path, keys=None) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in (keys or z.files)}


def clip(shard: dict, i: int) -> dict:
    return {k: v[i] for k, v in shard.items()}


def replay_in_mujoco(env, c: dict, actions=None) -> np.ndarray:
    """Run MuJoCo from the clip's exact initial state with the given actions
    (default: the clip's own). Returns block positions (T+1, 8, 3), unused slots zero."""
    from .hand_env import HandEnv  # noqa: F401  (env is a HandEnv)
    actions = c["action"] if actions is None else actions
    env.reset(n_blocks=int(c["n_blocks"]), init_qpos=c["init_qpos"], init_qvel=c["init_qvel"])
    n = env.n_blocks
    out = np.zeros((len(actions) + 1, 8, 3))
    out[0, :n] = env.all_block_pos()[:n]
    for k, a in enumerate(actions):
        env.action = np.asarray(a, dtype=np.float64)
        env._apply(env.action)
        env._physics_tick()
        out[k + 1, :n] = env.all_block_pos()[:n]
    return out
