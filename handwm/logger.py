"""Per-physics-step episode logging to logs/*.npz (+ a readable .csv)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Columns written to the CSV (the .npz has everything, including full qpos/qvel).
CSV_COLUMNS = [
    ("t", 1), ("tick", 1),
    ("in_x", 1), ("in_y", 1), ("in_yaw", 1), ("in_bend", 1),
    ("in_data_age", 1), ("in_valid", 1), ("accepted", 1),
    ("cmd", 4), ("finger_q", 4), ("tip_pos", 3),
    ("block_pos", 3), ("block_quat", 4), ("block_linvel", 3), ("block_angvel", 3),
]


class EpisodeLogger:
    def __init__(self, out_dir: str | Path, meta: dict, csv_columns=CSV_COLUMNS):
        self.out_dir = Path(out_dir)
        self.meta = dict(meta)
        self.csv_columns = csv_columns  # None = no .csv
        self.rows: dict[str, list] = {}

    def __len__(self):
        return len(self.rows.get("t", []))

    def append(self, **fields):
        for k, v in fields.items():
            self.rows.setdefault(k, []).append(np.asarray(v, dtype=np.float64))

    def save(self, tag: str = "ep") -> Path | None:
        if len(self) == 0:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{tag}_{stamp}_{int(time.time() * 1000) % 1000:03d}.npz"
        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)
        if self.csv_columns:
            self._write_csv(path.with_suffix(".csv"), arrays, self.csv_columns)
        return path

    @staticmethod
    def _write_csv(path: Path, arrays: dict, columns):
        cols, header = [], []
        for name, width in columns:
            a = arrays[name].reshape(len(arrays["t"]), -1)
            cols.append(a)
            header += [name] if width == 1 else [f"{name}_{i}" for i in range(width)]
        np.savetxt(path, np.hstack(cols), delimiter=",", header=",".join(header),
                   comments="", fmt="%.6g")


def load_episode(path: str | Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    return meta, arrays


def latest_episode(log_dir: str | Path) -> Path:
    files = sorted(Path(log_dir).glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no .npz episodes in {log_dir}")
    return files[-1]
