"""Convert samp830 LIBERO rollouts into canonical episodes.

The source is a LeRobot-style directory with one Parquet file per episode and
an ``episode_manifest.jsonl`` sidecar.  It has no reward column, so the sparse
LIBERO task reward is reconstructed: ``1`` on the final transition of a
successful episode and ``0`` elsewhere.  A task horizon of 500 marks the
remaining-step countdown; failure episodes are terminal by horizon exhaustion.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .adapters import load_rlinf, write_dataset

SOURCE_ID = "samp830-libero40-v1"
SOURCE_REVISION = "71a88ad0eaa5f5dc5237e6ede56e7cc3d4a3e9e8"
SOURCE_MANIFEST = "episode_manifest.jsonl"
DEFAULT_TASK_HORIZON = 500
FEATURE_VERSION = "libero-state8-v1"
SUITE_BY_TASK = {
    **{i: "libero_10" for i in range(0, 10)},
    **{i: "libero_goal" for i in range(10, 20)},
    **{i: "libero_object" for i in range(20, 30)},
    **{i: "libero_spatial" for i in range(30, 40)},
}


def read_manifest(source_root):
    path = Path(source_root) / SOURCE_MANIFEST
    if not path.is_file():
        raise ValueError(f"missing {path}")
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        eid = int(row["episode_index"])
        if eid in records:
            raise ValueError(f"duplicate episode_index {eid}")
        if int(row["task_index"]) not in SUITE_BY_TASK:
            raise ValueError(f"episode {eid}: task_index outside 0..39")
        if not isinstance(row["success"], bool):
            raise ValueError(f"episode {eid}: success must be boolean")
        if int(row["length"]) < 1:
            raise ValueError(f"episode {eid}: nonpositive length")
        records[eid] = row
    if not records:
        raise ValueError("empty episode manifest")
    return records


def suite_for_task(task_index):
    if task_index not in SUITE_BY_TASK:
        raise ValueError("task_index must be between 0 and 39")
    return SUITE_BY_TASK[task_index]


def episode_ids_for_task(records, task_index):
    ids = sorted(eid for eid, row in records.items()
                 if int(row["task_index"]) == int(task_index))
    if not ids:
        raise ValueError(f"task_index {task_index} has no episodes")
    return ids


def _episode_path(source_root, episode_id):
    return Path(source_root) / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"


def build_reward_sidecar(source_root, records, episode_ids, output_path):
    episode_parts, frame_parts, reward_parts = [], [], []
    for eid in episode_ids:
        path = _episode_path(source_root, eid)
        if not path.is_file():
            raise ValueError(f"missing episode file {path}")
        table = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
        episodes = table.column("episode_index").to_numpy(zero_copy_only=False)
        frames = table.column("frame_index").to_numpy(zero_copy_only=False)
        tasks = table.column("task_index").to_numpy(zero_copy_only=False)
        if not np.all(tasks == int(records[eid]["task_index"])):
            raise ValueError(f"episode {eid}: task_index disagrees with manifest")
        if len(frames) != int(records[eid]["length"]):
            raise ValueError(f"episode {eid}: length mismatch")
        if not np.all(episodes == eid):
            raise ValueError(f"episode {eid}: mixed episode_index values")
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"episode {eid}: frame_index must start at zero and be contiguous")
        reward = np.zeros(len(frames), dtype=np.float32)
        if records[eid]["success"]:
            reward[-1] = 1.0
        episode_parts.append(np.full(len(frames), eid, dtype=np.int64))
        frame_parts.append(frames.astype(np.int64, copy=False))
        reward_parts.append(reward)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "episode_index": pa.array(np.concatenate(episode_parts), type=pa.int64()),
        "frame_index": pa.array(np.concatenate(frame_parts), type=pa.int64()),
        "reward": pa.array(np.concatenate(reward_parts), type=pa.float32()),
    })
    pq.write_table(table, output_path)
    return {"rows": table.num_rows, "episodes": len(episode_ids)}


def stage_task(source_root, episode_ids, work_root):
    data_dir = Path(work_root) / "data/chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    for eid in episode_ids:
        source = _episode_path(source_root, eid).resolve()
        if not source.is_file():
            raise ValueError(f"missing episode file {source}")
        link = data_dir / f"episode_{eid:06d}.parquet"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(source, link)
    return data_dir


def make_config(task_index, task_horizon=DEFAULT_TASK_HORIZON, state_columns=("state",),
                action_column="actions", source_id=SOURCE_ID, policy_id=None,
                source_revision=SOURCE_REVISION, source_manifest=SOURCE_MANIFEST):
    suite = suite_for_task(task_index)
    return {
        "source_id": source_id,
        "source_revision": source_revision,
        "source_manifest": source_manifest,
        "policy_id": policy_id or f"samp830/flow_bc/{suite}",
        "task_id": f"{suite}/task_{task_index}",
        "task_horizon": int(task_horizon),
        "state_columns": list(state_columns),
        "action_column": action_column,
        "reward_column": "reward",
        "reward_sidecar": "meta/returns_samp830.parquet",
        "episode_column": "episode_index",
        "frame_column": "frame_index",
        "rows_are_primitive_transitions": True,
        "episode_end_is_terminal": True,
        "feature_version": FEATURE_VERSION,
        "data_glob": "data/**/*.parquet",
    }


def _prepare_work_dir(work_root, overwrite_work):
    work_root = Path(work_root)
    if work_root.exists() and any(work_root.iterdir()):
        if not overwrite_work:
            raise ValueError("work dir must be empty; pass overwrite_work=True to replace generated staging")
        for name in ("data", "meta"):
            path = work_root / name
            if path.exists():
                shutil.rmtree(path)
        config = work_root / "config.json"
        if config.exists():
            config.unlink()
    work_root.mkdir(parents=True, exist_ok=True)


def _enrich_episodes(episodes, records, source_root, source_revision):
    for episode in episodes:
        eid = int(episode.episode_id.split(":")[-1])
        row = records[eid]
        episode.metadata.update({
            "source_root": str(Path(source_root).resolve()),
            "source_revision": source_revision,
            "source_manifest": SOURCE_MANIFEST,
            "success": bool(row["success"]),
            "length": int(row["length"]),
            "raw_demo": row.get("raw_demo"),
            "task_file": row.get("task_file"),
            "task": row.get("task"),
            "suite": row.get("suite"),
            "reward_reconstruction": "sparse task success on final transition",
        })


def _validate_episodes(episodes, records, task_horizon):
    if len(episodes) != len(records):
        raise ValueError("converted episode count does not match manifest")
    for episode in episodes:
        eid = int(episode.episode_id.split(":")[-1])
        row = records[eid]
        if len(episode) != int(row["length"]):
            raise ValueError(f"episode {eid}: converted length mismatch")
        if not row["success"] and len(episode) != int(task_horizon):
            raise ValueError(f"episode {eid}: failed rollout ends before task horizon; cannot assume true termination")
        expected = 1.0 if row["success"] else 0.0
        if not np.isclose(float(episode.reward.sum()), expected):
            raise ValueError(f"episode {eid}: reconstructed reward mismatch")
        if not episode.terminated[-1] or episode.truncated[-1]:
            raise ValueError(f"episode {eid}: final boundary must be terminated, not truncated")
        if int(episode.remaining_steps[0]) != int(task_horizon):
            raise ValueError(f"episode {eid}: first remaining step mismatch")
        if int(episode.remaining_steps[-1]) != int(task_horizon) - (len(episode) - 1):
            raise ValueError(f"episode {eid}: remaining step countdown mismatch")


def _split_outcomes(manifest, records):
    result = {}
    for name, ids in manifest["split"].items():
        successes = sum(bool(records[int(eid.split(":")[-1])]["success"]) for eid in ids)
        result[name] = {"episodes": len(ids), "success": successes,
                        "failure": len(ids) - successes}
    return result


def convert_task(source_root, task_index, output_root, work_root, seed=0,
                 task_horizon=DEFAULT_TASK_HORIZON, state_columns=("state",),
                 action_column="actions", source_id=SOURCE_ID, policy_id=None,
                 source_revision=SOURCE_REVISION, overwrite_work=False):
    task_index = int(task_index)
    records = read_manifest(source_root)
    episode_ids = episode_ids_for_task(records, task_index)
    suite = suite_for_task(task_index)
    _prepare_work_dir(work_root, overwrite_work)
    stage_task(source_root, episode_ids, work_root)
    build_reward_sidecar(source_root, records, episode_ids,
                         Path(work_root) / "meta/returns_samp830.parquet")
    config = make_config(task_index, task_horizon, state_columns, action_column,
                         source_id, policy_id, source_revision)
    (Path(work_root) / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    episodes = load_rlinf(work_root, config)
    task_records = {eid: records[eid] for eid in episode_ids}
    _enrich_episodes(episodes, task_records, source_root, source_revision)
    _validate_episodes(episodes, task_records, task_horizon)
    manifest = write_dataset(episodes, output_root, seed)
    split_outcomes = _split_outcomes(manifest, task_records)
    warnings = [f"{name}: missing success or failure"
                for name, counts in split_outcomes.items()
                if counts["success"] == 0 or counts["failure"] == 0]
    return {
        "episodes": len(episodes),
        "task_index": task_index,
        "suite": suite,
        "task_id": config["task_id"],
        "policy_id": config["policy_id"],
        "task_horizon": int(task_horizon),
        "source_id": source_id,
        "source_revision": source_revision,
        "split_seed": int(seed),
        "split_sizes": {name: len(ids) for name, ids in manifest["split"].items()},
        "split_outcomes": split_outcomes,
        "warnings": warnings,
        "output": str(Path(output_root)),
        "work_dir": str(Path(work_root)),
        "config_path": str(Path(work_root) / "config.json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert samp830 LIBERO tasks to canonical NPZ")
    parser.add_argument("source_root")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-horizon", type=int, default=DEFAULT_TASK_HORIZON)
    parser.add_argument("--overwrite-work", action="store_true")
    args = parser.parse_args(argv)
    summary = convert_task(args.source_root, args.task_index, args.output, args.work_dir,
                           seed=args.seed, task_horizon=args.task_horizon,
                           overwrite_work=args.overwrite_work)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
