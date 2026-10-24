"""Hand motion WITHOUT blocks, recomputed from a clip's actions.

In the full simulation the hand is occasionally pushed by the blocks (e.g. a palm-down hand
resting on a block is lifted ~5 cm). Those hand keypoints would quietly tell a world model
where the blocks are - information it will not have when you play, because then the hand comes
from your actions alone. So the world model's hand input is recomputed here by replaying each
clip's actions in a simulation with every block parked far away. Same servos, same timing,
fully deterministic - exactly what the live play mode does.
"""
from __future__ import annotations

import numpy as np

from .hand_scene import BLOCK_HALF, MAX_BLOCKS


def replay_hand_only(env, init_qpos, actions):
    """env: a HandEnv (any blocks). Returns keypoints (T+1, 21, 3) and hand_pose (T+1, 5)."""
    q = np.array(init_qpos, dtype=np.float64).copy()
    for i in range(MAX_BLOCKS):
        a = env._bq[i]
        q[a:a + 3] = [2 + 0.2 * i, 2, BLOCK_HALF]
        q[a + 3:a + 7] = [1, 0, 0, 0]
    env.reset(n_blocks=0, init_qpos=q, init_qvel=np.zeros(env.model.nv))
    kps, poses = [env.keypoints()], [env.data.qpos[env._hq].copy()]
    for a in actions:
        env.action = np.asarray(a, dtype=np.float64)
        env._apply(env.action)
        env._physics_tick()
        kps.append(env.keypoints())
        poses.append(env.data.qpos[env._hq].copy())
    return np.array(kps, np.float32), np.array(poses, np.float32)
