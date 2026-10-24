"""Loading the generated clips as state sequences for the coordinate world model.

Block state (13 numbers per block): position (3), orientation quaternion w,x,y,z (4),
linear velocity (3), angular velocity (3) - all world frame. By default the velocities are
"change over the last 0.05 s step / 0.05 s" (finite difference) - the same definition the
world model uses for its own predictions - instead of MuJoCo's instantaneous velocities.

Splits are by clip (never by time step). clip_index % 20 == 0 -> test, == 1 -> val, else train.
Clips with more than `max_train_blocks` blocks are never trained on; the test-split ones form
the out-of-distribution test set ("test_ood", e.g. 6-8 blocks).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import list_shards

KEYS = ["block_pos", "block_quat", "block_vel", "keypoints", "action", "hand_pose", "n_blocks"]
STATE_DIM = 13


def canon_quat(q: np.ndarray) -> np.ndarray:
    """q and -q are the same rotation; keep w >= 0 so the model sees one form."""
    return np.where(q[..., :1] < 0, -q, q)


DT = 0.05


def quat_to_rotvec(q):
    q = canon_quat(q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12))
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., :1])
    return np.where(s > 1e-9, v / np.maximum(s, 1e-12) * ang, 2 * v)


def quat_mul_np(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def finite_diff_velocity(pos, quat, prev_pos, prev_quat):
    """Linear / angular velocity over one step: (pos - prev) / DT, rotvec(q * prev^-1) / DT."""
    conj = prev_quat * np.array([1, -1, -1, -1])
    return (pos - prev_pos) / DT, quat_to_rotvec(quat_mul_np(quat, conj)) / DT


def load_states(data_dir, max_shards: int | None = None, verbose: bool = True,
                hand_source: str = "hand_only", velocity: str = "finite_diff") -> dict:
    """hand_source="hand_only": hand keypoints/pose replayed WITHOUT blocks (what the model may
    use - see hand_only.py; needs scripts/prep_hand_only.py). "sim": as recorded in the full
    simulation (can leak block information through the hand)."""
    shards = list_shards(data_dir)[:max_shards]
    if not shards:
        raise FileNotFoundError(f"no shards in {data_dir}")
    parts = {k: [] for k in KEYS}
    idx = []
    for s, p in enumerate(shards):
        with np.load(p) as z:  # only these members are decompressed (frames are skipped)
            for k in KEYS:
                parts[k].append(z[k])
            n = len(z["n_blocks"])
        if hand_source == "hand_only":
            ho = p.with_name(p.name.replace("shard_", "handonly_"))
            if not ho.exists():
                raise FileNotFoundError(f"{ho.name} missing - run scripts/prep_hand_only.py first")
            with np.load(ho) as z:
                parts["keypoints"][-1], parts["hand_pose"][-1] = z["keypoints"], z["hand_pose"]
        shard_no = int(p.stem.split("_")[1])
        idx.append(shard_no * n + np.arange(n))
        if verbose and (s + 1) % 50 == 0:
            print(f"  loaded {s + 1}/{len(shards)} shards", flush=True)
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["clip_index"] = np.concatenate(idx)
    active = np.arange(8)[None, :] < d["n_blocks"].astype(int)[:, None]       # (clips, 8)
    q = canon_quat(d.pop("block_quat"))
    act4 = active[:, None, :, None]                                          # (clips, 1, 8, 1)
    q = np.where(act4, q, np.array([1, 0, 0, 0], q.dtype))  # unused slots: identity quaternion
    p = d.pop("block_pos")
    vel = d.pop("block_vel")
    if velocity == "finite_diff":   # step 0: blocks start at rest
        vel = np.zeros_like(vel)
        lv, av = finite_diff_velocity(p[:, 1:], q[:, 1:], p[:, :-1], q[:, :-1])
        vel[:, 1:, :, :3], vel[:, 1:, :, 3:] = lv, av
        vel = np.where(act4, vel, 0)
    d["blocks"] = np.concatenate([p, q, vel], -1).astype(np.float32)
    d["keypoints"] = d["keypoints"].astype(np.float32)
    d["action"] = d["action"].astype(np.float32)
    d["hand_pose"] = d["hand_pose"].astype(np.float32)
    nb = d["n_blocks"].astype(int)
    d["mask"] = (np.arange(8)[None, :] < nb[:, None])
    return d


def excluded_clips(data_dir) -> list[int]:
    """Global clip indices to leave out of every split: the clips listed in
    <data>/deep_contacts.json (flagged 'unstable' + any with a true contact deeper than 5 mm).
    Decided with Xia on 2026-10-19. Empty if the file does not exist."""
    import json
    p = Path(data_dir) / "deep_contacts.json"
    if not p.exists():
        return []
    info = json.loads((Path(data_dir) / "dataset_info.json").read_text(encoding="utf-8"))
    dc = json.loads(p.read_text(encoding="utf-8"))
    per = int(info["per_shard"])
    clips = {s * per + i for s, i in dc["unstable"]} | {e[0] * per + e[1] for e in dc["events"]}
    return sorted(int(c) for c in clips)


def split_indices(d: dict, max_train_blocks: int = 5, exclude=()) -> dict[str, np.ndarray]:
    ci, nb = d["clip_index"], d["n_blocks"].astype(int)
    part = ci % 20
    in_dist = nb <= max_train_blocks
    keep = ~np.isin(ci, np.asarray(list(exclude), dtype=ci.dtype))
    return {
        "train": np.flatnonzero((part >= 2) & in_dist & keep),
        "val": np.flatnonzero((part == 1) & in_dist & keep),
        "test": np.flatnonzero((part == 0) & in_dist & keep),
        "test_ood": np.flatnonzero((part == 0) & ~in_dist & keep),
    }


def subset(d: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in d.items()}


class WindowSampler:
    """Random training windows: K+1 consecutive states of one clip."""

    def __init__(self, d: dict, K: int, rng: np.random.Generator):
        self.d, self.K, self.rng = d, K, rng
        self.N, self.T1 = d["blocks"].shape[:2]

    def sample(self, B: int) -> dict:
        c = self.rng.integers(self.N, size=B)
        t = self.rng.integers(self.T1 - self.K, size=B)
        tt = t[:, None] + np.arange(self.K + 1)[None]
        cc = c[:, None]
        return {
            "blocks": self.d["blocks"][cc, tt],        # (B, K+1, 8, 13)
            "keypoints": self.d["keypoints"][cc, tt],  # (B, K+1, 21, 3)
            "hand_pose": self.d["hand_pose"][cc, tt],  # (B, K+1, 5)
            "action": self.d["action"][cc, tt[:, :-1]],  # (B, K, 9)
            "mask": self.d["mask"][c],                 # (B, 8)
        }
