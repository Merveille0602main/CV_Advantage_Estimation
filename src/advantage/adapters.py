"""Explicit LeRobot/RLinf Parquet mapping; no dependency on the RLinf runtime."""
from pathlib import Path
import json
import numpy as np
from .schema import Episode


def _rows(paths, columns):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError('Install the rlinf extra: pip install -e ".[rlinf]"') from exc
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for batch in parquet.iter_batches(columns=sorted(set(columns)), batch_size=8192):
            yield from batch.to_pylist()


def inspect_parquet(root):
    import pyarrow.parquet as pq
    paths = sorted(Path(root).glob("data/**/*.parquet"))
    if not paths:
        raise ValueError("no data/**/*.parquet files; inspect the dataset's actual format")
    first = pq.ParquetFile(paths[0])
    return {"files": len(paths), "first_file": str(paths[0]),
            "schema": str(first.schema_arrow), "rows_first_file": first.metadata.num_rows}


def load_rlinf(root, config):
    """Load low-dimensional states from LeRobot data and raw reward sidecars.

    Config must explicitly declare boundary semantics and primitive-step units.
    This adapter does NOT use precomputed returns/advantages as training labels.
    Source-specific reward/action timing still requires auditing a real sample.
    """
    root = Path(root)
    required = ("source_id", "policy_id", "task_id", "task_horizon", "state_columns",
                "action_column", "reward_column", "rows_are_primitive_transitions")
    for k in required:
        if k not in config:
            raise ValueError(f"mapping requires {k}")
    if config["rows_are_primitive_transitions"] is not True:
        raise ValueError("expand action chunks and align rewards before importing")
    if not config["state_columns"] or int(config["task_horizon"]) < 1:
        raise ValueError("state_columns and a positive task_horizon are required")
    ep_key, frame_key = config.get("episode_column", "episode_index"), config.get("frame_column", "frame_index")
    reward_key = config["reward_column"]
    cols = [ep_key, frame_key, config["action_column"], *config["state_columns"]]
    for name in ("terminal_column", "truncated_column", "parent_column"):
        if config.get(name):
            cols.append(config[name])
    if not config.get("terminal_column") and config.get("episode_end_is_terminal") is not True:
        raise ValueError("declare a terminal_column or explicitly confirm episode_end_is_terminal")
    next_cols = config.get("next_state_columns")
    if next_cols:
        cols.extend(next_cols)
    sidecar = config.get("reward_sidecar")
    rewards = {}
    if sidecar:
        paths = sorted(root.glob(sidecar))
        if not paths:
            raise ValueError("reward_sidecar matched no files")
        for row in _rows(paths, [ep_key, frame_key, reward_key]):
            key = (row[ep_key], row[frame_key])
            if key in rewards:
                raise ValueError(f"duplicate reward sidecar key {key}")
            rewards[key] = row[reward_key]
    else:
        cols.append(reward_key)
    paths = sorted(root.glob(config.get("data_glob", "data/**/*.parquet")))
    if not paths:
        raise ValueError("no data files matched")
    grouped = {}
    for row in _rows(paths, cols):
        if sidecar:
            key = (row[ep_key], row[frame_key])
            if key not in rewards:
                raise ValueError(f"missing reward for {key}")
            row[reward_key] = rewards[key]
        grouped.setdefault(row[ep_key], []).append(row)
    episodes = []
    for raw_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r[frame_key])
        frames = np.array([r[frame_key] for r in rows])
        if frames[0] != 0 or not np.array_equal(frames, np.arange(len(rows))):
            raise ValueError(f"episode {raw_id}: missing/duplicate/nonzero-start frame indices")
        def features(keys):
            return np.stack([np.concatenate([np.asarray(r[k], dtype=float).reshape(-1) for k in keys]) for r in rows])
        obs = features(config["state_columns"])
        term = np.array([r[config["terminal_column"]] for r in rows]) if config.get("terminal_column") else np.zeros(len(rows), bool)
        trunc = np.array([r[config["truncated_column"]] for r in rows]) if config.get("truncated_column") else np.zeros(len(rows), bool)
        for flags in (term, trunc):
            if flags.dtype != bool:
                raise ValueError("source boundary columns must be booleans")
        if config.get("episode_end_is_terminal"):
            if trunc[-1]:
                raise ValueError("terminal declaration conflicts with truncated source row")
            term[-1] = True
        if next_cols:
            nxt = features(next_cols)
        else:
            if not term[-1]:
                raise ValueError("truncated episode requires a real final next observation")
            # A terminal placeholder is never bootstrapped; it is not a physical state.
            nxt = np.concatenate([obs[1:], np.zeros_like(obs[:1])])
        parent = rows[0][config["parent_column"]] if config.get("parent_column") else raw_id
        if config.get("parent_column") and any(r[config["parent_column"]] != parent for r in rows):
            raise ValueError("parent identity changes inside an episode")
        prefix = config["source_id"]
        episodes.append(Episode(
            episode_id=f"{prefix}:{raw_id}", parent_id=f"{prefix}:{parent}",
            task_id=str(config["task_id"]), policy_id=config["policy_id"],
            obs=obs, next_obs=nxt,
            action=np.stack([np.asarray(r[config["action_column"]], float).reshape(-1) for r in rows]),
            reward=np.array([r[reward_key] for r in rows], float),
            terminated=term, truncated=trunc,
            remaining_steps=int(config["task_horizon"])-frames,
            metadata={"schema_version": 1, "source_format": "LeRobot-Parquet/RLinf",
                      "feature_version": config.get("feature_version", "lowdim-v1"),
                      "mapping": config, "source_root": str(root.resolve()),
                      "family_identity_verified": bool(config.get("parent_column")),
                      "terminal_next_obs_placeholder": not bool(next_cols)}))
    return episodes


def write_dataset(episodes, destination, seed=0):
    from .schema import split_families, file_hash
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("destination must be empty; do not overwrite a frozen dataset")
    destination.mkdir(parents=True, exist_ok=True)
    split = split_families(episodes, seed)
    records = []
    for i, ep in enumerate(episodes):
        name = f"episode_{i:06d}.npz"
        ep.save(destination / name)
        records.append({"episode_id": ep.episode_id, "parent_id": ep.parent_id,
                        "path": name, "sha256": file_hash(destination/name), "transitions": len(ep)})
    manifest = {"schema_version": 1, "split_seed": seed, "split": split, "episodes": records}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
