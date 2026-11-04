"""Canonical transition-aligned episodes. No pickle or implicit reward inference."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import numpy as np

ARRAY_FIELDS = ("obs", "next_obs", "action", "reward", "terminated", "truncated", "remaining_steps")


@dataclass
class Episode:
    episode_id: str
    parent_id: str
    task_id: str
    policy_id: str
    obs: np.ndarray
    next_obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    remaining_steps: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ARRAY_FIELDS:
            setattr(self, key, np.asarray(getattr(self, key)))
        self.validate()

    def __len__(self):
        return len(self.reward)

    def validate(self):
        for key in ("episode_id", "parent_id", "task_id", "policy_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty string")
        t = len(self.reward)
        if t == 0:
            raise ValueError("empty episode")
        for key in ARRAY_FIELDS:
            a = getattr(self, key)
            if a.ndim == 0 or len(a) != t:
                raise ValueError(f"unaligned field: {key}")
            if not np.issubdtype(a.dtype, np.number) and a.dtype != bool:
                raise ValueError(f"nonnumeric field: {key}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"nonfinite field: {key}")
        if self.obs.ndim != 2 or self.obs.shape != self.next_obs.shape or self.action.ndim != 2:
            raise ValueError("obs/next_obs must be matching feature matrices; action must be a matrix")
        for key in ("reward", "terminated", "truncated", "remaining_steps"):
            if getattr(self, key).shape != (t,):
                raise ValueError(f"{key} must be a vector")
        for key in ("terminated", "truncated"):
            if getattr(self, key).dtype != bool:
                raise ValueError(f"{key} must be boolean, not an ambiguous done array")
        if np.any(self.terminated & self.truncated):
            raise ValueError("a transition cannot be both terminated and truncated")
        if np.any((self.terminated | self.truncated)[:-1]):
            raise ValueError("an episode cannot contain an internal boundary")
        if not (self.terminated[-1] or self.truncated[-1]):
            raise ValueError("last transition must declare termination or truncation")
        if not np.issubdtype(self.remaining_steps.dtype, np.integer):
            raise ValueError("remaining_steps must be integer")
        if np.any(self.remaining_steps < 1) or np.any(np.diff(self.remaining_steps) != -1):
            raise ValueError("remaining_steps must count down in primitive control steps")
        if t > 1 and not np.allclose(self.next_obs[:-1], self.obs[1:], rtol=1e-6, atol=1e-7):
            raise ValueError("next_obs is not aligned with the following observation")
        if self.truncated[-1] and self.remaining_steps[-1] <= 1:
            raise ValueError("task horizon exhaustion is terminal, not collector truncation")
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {k: getattr(self, k) for k in ("episode_id", "parent_id", "task_id", "policy_id", "metadata")}
        np.savez_compressed(path, header=json.dumps(header, sort_keys=True),
                            **{k: getattr(self, k) for k in ARRAY_FIELDS})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(**json.loads(str(z["header"])), **{k: z[k] for k in ARRAY_FIELDS})


def split_families(episodes, seed=0, fractions=(0.7, 0.1, 0.2)):
    """Assign entire parent families. Tiny datasets may yield empty splits."""
    if len(fractions) != 3 or min(fractions) < 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("expected three nonnegative fractions summing to one")
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("duplicate episode_id")
    parents = sorted({e.parent_id for e in episodes})
    np.random.default_rng(seed).shuffle(parents)
    a, b = int(len(parents)*fractions[0]), int(len(parents)*sum(fractions[:2]))
    mapping = {p: s for s, group in zip(("train", "validation", "test"),
               (parents[:a], parents[a:b], parents[b:])) for p in group}
    return {s: [e.episode_id for e in episodes if mapping[e.parent_id] == s]
            for s in ("train", "validation", "test")}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
