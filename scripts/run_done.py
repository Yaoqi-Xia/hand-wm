"""Exit 0 if a COMPLETED training run with this tag exists, else 1 (used by launchers to skip
finished work). Complete = runs/train_*_<tag>/ has model.pt, model_onestep.pt and a log whose
last row reached the planned number of steps of the last stage.

  python scripts/run_done.py main_s1
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def complete(d: Path) -> bool:
    try:
        s = json.loads((d / "settings.json").read_text(encoding="utf-8"))
        rows = list(csv.DictReader(open(d / "log.csv", encoding="utf-8")))
    except (OSError, ValueError):
        return False
    if not ((d / "model.pt").exists() and (d / "model_onestep.pt").exists() and rows):
        return False
    stage, steps = (2, s["stage2_steps"]) if s["stage2_steps"] > 0 else (1, s["stage1_steps"])
    last = rows[-1]
    return int(last["stage"]) == stage and int(last["step"]) == steps


if __name__ == "__main__":
    tag = sys.argv[1]
    done = [d for d in (ROOT / "runs").glob(f"train_*_{tag}") if complete(d)]
    print(f"{tag}: {'already complete -> skip (' + done[-1].name + ')' if done else 'not done yet'}")
    sys.exit(0 if done else 1)
