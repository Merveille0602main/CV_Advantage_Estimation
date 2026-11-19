"""Common protocol and paths for the three-task experiment."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("STUDY_ROOT", "/mnt/d/CV/training_runs/minerva_three/run01"))
OLD = Path(os.environ.get("MINERVA_FIRST_RUN", "/mnt/d/CV/training_runs/minerva_goal13/20261119_run01"))
WEIGHTS = Path(os.environ.get("MINERVA_WEIGHTS", "/mnt/d/CV/minerva_smoke/inputs/t05_0.54M"))
REV = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
POLICY = "MINERVA-t05_0.54M-38e7414-temp0.85-ensemble0.01-step1"
TASKS = {
    13: ("libero_goal", 6, "put the cream cheese in the bowl", 300),
    19: ("libero_goal", 0, "open the middle drawer of the cabinet", 300),
    25: ("libero_object", 7, "pick up the milk and place it in the basket", 280),
}
NAMES = ("recorded", "resampled", "demo", "mild", "strong")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf8")
    temp.replace(path)


def save_episode(path, ep):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.npz")
    ep.save(temp)
    temp.replace(path)


def task_dir(task):
    return ROOT / str(task)


def base_dir(task):
    return OLD if task == 13 else task_dir(task) / "base"


def split():
    ids = np.random.default_rng(12).permutation(50).tolist()
    return dict(train=sorted(ids[:35]), validation=sorted(ids[35:40]), test=sorted(ids[40:]))


def eid(init, rep=0):
    return f"init{init:02d}_rep{rep}"
