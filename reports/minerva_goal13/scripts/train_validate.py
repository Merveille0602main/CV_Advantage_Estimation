"""Freeze the 100 completed episodes, fit six upstream checkpoints, report validation RMSE."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from advantage.cli import read_split
from advantage.schema import Episode, file_hash
from advantage.qvalue import QPredictor
from advantage.targets import discounted_returns
from advantage.value import ValuePredictor

GAMMA = 0.99
RELEASE = "7671b454a281506a68706e7b57660742b7da85da"
BASE = Path("/mnt/d/CV")
WEIGHTS = BASE / "minerva_smoke/inputs/t05_0.54M"
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "advantage_studies_src"


def write_json(path, obj, *, frozen=False):
    content = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if frozen and path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Frozen run inputs changed: {path}")
        return
    temp = path.with_suffix(".tmp")
    temp.write_text(content)
    os.replace(temp, path)


def verify_snapshot(path, frame, policy_seed):
    with np.load(path, allow_pickle=False) as z:
        assert int(z["frame"]) == frame and int(z["policy_seed"]) == policy_seed
        assert z["mujoco_state"].ndim == 1 and len(z["mujoco_state"]) > 25
        assert np.isfinite(z["mujoco_state"]).all() and int(z["mj_state_spec"]) > 0
        assert z["torch_rng"].size > 0 and z["np_rng_keys"].size > 0
        assert "python_rng" in z and "controller_goal_pos" in z
        for view in ("image", "image2"):
            assert z["current_" + view].shape == (360, 360, 3)
        if frame:
            for key in ("queue_observation.state", "queue_observation.images",
                        "ensembled_actions", "ensembled_actions_count"):
                assert np.isfinite(z[key]).all() and z[key].size > 0
    return file_hash(path)


def freeze_dataset(root):
    ids = np.random.default_rng(12).permutation(50)
    families = {k: sorted(map(int, subset)) for k, subset in zip(
        ("train", "validation", "test"), (ids[:35], ids[35:40], ids[40:]))}
    membership = {k: [] for k in families}
    records, snapshots, stats = [], {}, {}
    initial_hashes, policy_ids = set(), set()
    for split, indices in families.items():
        successes = steps = 0
        for init in indices:
            first = None
            for rep in range(2):
                name = f"init{init:02d}_rep{rep}"
                path = root / "episodes" / (name + ".npz")
                ep = Episode.load(path)
                meta = ep.metadata
                assert ep.episode_id == name and ep.parent_id == f"init{init:02d}"
                assert ep.task_id == "libero_goal_6" and meta["split"] == split and meta["init_id"] == init
                assert meta["env_seed"] == 11000 + init and meta["policy_seed"] == 21000 + 2*init + rep
                assert ep.obs.shape == (len(ep), 8) and ep.action.shape == (len(ep), 7)
                assert ep.remaining_steps[0] == 300 and len(ep) <= 300 and ep.terminated[-1]
                assert np.all(ep.reward[:-1] == 0) and ep.reward[-1] in (0, 1)
                success = bool(ep.reward[-1])
                assert meta["end_reason"] == ("success" if success else "time_limit")
                assert success or len(ep) == 300
                expected_frames = list(range(0, len(ep), 40)) if split == "test" else []
                assert meta["snapshot_frames"] == expected_frames
                for t in expected_frames:
                    rel = Path("snapshots") / name / f"{t:03d}.npz"
                    snapshots[str(rel)] = verify_snapshot(root / rel, t, meta["policy_seed"])
                if first is None:
                    first = ep
                    assert meta["init_sha256"] not in initial_hashes
                    initial_hashes.add(meta["init_sha256"])
                else:
                    assert meta["init_sha256"] == first.metadata["init_sha256"]
                    np.testing.assert_allclose(ep.obs[0], first.obs[0], atol=1e-6, rtol=1e-6)
                    if np.array_equal(ep.action, first.action):
                        raise ValueError(f"Duplicate complete action history at initial state {init}")
                policy_ids.add(ep.policy_id)
                membership[split].append(name)
                records.append(dict(episode_id=name, parent_id=ep.parent_id, path=str(path.relative_to(root)),
                                    sha256=file_hash(path), length=len(ep), **meta))
                successes += int(success)
                steps += len(ep)
        stats[split] = dict(families=len(indices), episodes=len(indices)*2, success=successes,
                            timeout=len(indices)*2-successes, transitions=steps)
    assert len(policy_ids) == 1 and len(initial_hashes) == 50
    assert len(list((root/"episodes").glob("*.npz"))) == 100
    assert set(str(p.relative_to(root)) for p in (root/"snapshots").rglob("*.npz")) == set(snapshots)
    manifest = dict(dataset="minerva_goal13", policy_id=policy_ids.pop(), source_commit=RELEASE,
                    families=families, split_seed=12, split=membership, episodes=records,
                    horizon=300, gamma=GAMMA, reward="successful terminal action=1; otherwise=0",
                    task="put the cream cheese in the bowl", feature_order="eef_pos3, axisangle3, gripper_qpos2")
    write_json(root/"manifest.json", manifest, frozen=True)
    write_json(root/"snapshot_hashes.json", snapshots, frozen=True)
    return stats, len(snapshots)


def rmse(pred, target):
    return float(np.sqrt(np.mean(np.square(pred-target))))


def main(root):
    stats, snapshot_count = freeze_dataset(root)
    train = read_split(root, "train")
    val = read_split(root, "validation")
    read_split(root, "test")  # Integrity check only.
    train_returns = np.concatenate([discounted_returns(e, GAMMA) for e in train])
    val_returns = np.concatenate([discounted_returns(e, GAMMA) for e in val])
    for ep in val:
        analytical = ep.reward[-1] * GAMMA ** np.arange(len(ep)-1, -1, -1)
        np.testing.assert_allclose(discounted_returns(ep, GAMMA), analytical)
    constant = float(train_returns.mean())
    # With many successful trajectories, elapsed time alone can predict discounted reward.
    # Use only the training trajectories still active at each particular control step.
    time_sum, time_count = np.zeros(300), np.zeros(300, dtype=int)
    for ep in train:
        time_sum[:len(ep)] += discounted_returns(ep, GAMMA)
        time_count[:len(ep)] += 1
    seen = time_count > 0
    time_mean = np.interp(np.arange(300), np.flatnonzero(seen), time_sum[seen]/time_count[seen])
    time_pred = np.concatenate([time_mean[:len(ep)] for ep in val])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provenance = dict(source_commit=RELEASE, manifest_sha256=file_hash(root/"manifest.json"),
                      minerva_code_sha256=file_hash(WEIGHTS.parent/"MINERVA-64c3cc8.tar.gz"),
                      weight_files={p.name: file_hash(p) for p in WEIGHTS.iterdir() if p.is_file()},
                      device=device, torch=str(torch.__version__), numpy=np.__version__,
                      policy_inference="CPU, OSMesa, n_action_steps=1, ensemble=0.01, noise_temperature=0.85",
                      gamma=GAMMA, updates=1000, hidden=[256,256], batch_size=128, lr=0.0003,
                      seeds=[0,1,2], collector_sha256=file_hash(HERE/"collect.py"))
    write_json(root/"provenance.json", provenance, frozen=True)
    details, summary = {}, {}
    models = root/"models"
    models.mkdir(exist_ok=True)
    for kind, cls, command in (("V", ValuePredictor, "train"), ("Q", QPredictor, "train-q")):
        values = []
        for seed in (0, 1, 2):
            name = f"{kind}_seed{seed}"
            dest = models / (name+".pt")
            if not dest.exists():
                import tempfile
                # Upstream CLI requires a fresh path, including after an interrupted run.
                with tempfile.TemporaryDirectory(dir=models) as tmp:
                    temp = Path(tmp)/"trained.pt"
                    args = [sys.executable, "-m", "advantage.cli", command, str(root),
                            "--output", str(temp), "--gamma", str(GAMMA), "--updates", "1000",
                            "--seed", str(seed), "--device", device]
                    if kind == "V":
                        args += ["--method", "mc"]
                    subprocess.run(args, check=True)
                    os.replace(temp, dest)
            predictor = cls(dest)
            config = predictor.checkpoint
            assert config["seed"] == seed and config["gamma"] == GAMMA and config["updates"] == 1000
            assert config["input_dim"] == (9 if kind == "V" else 16)
            assert config.get("method", config.get("kind")) == ("mc" if kind=="V" else "mc_q")
            assert np.isfinite(config["losses"]).all() and len(config["losses"]) == 1000
            pred = np.concatenate([predictor.predict(e) for e in val])
            assert pred.shape == val_returns.shape and np.isfinite(pred).all()
            error = rmse(pred, val_returns)
            details[name] = dict(sha256=file_hash(dest), validation_rmse=error)
            print(f"{name} validation RMSE {error:.6f}", flush=True)
            values.append(error)
        summary[kind] = dict(mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)),
                             by_seed=values)
    crashes = sum(len(p.read_text().splitlines()) for p in root.glob("crashes_worker*.jsonl"))
    report = dict(split=stats, checkpoints=details, validation=summary, snapshots=snapshot_count,
                  collection_errors_logged=crashes, train_return_mean=constant,
                  validation_constant_rmse=rmse(constant, val_returns), validation_unit="per transition",
                  validation_time_only_rmse=rmse(time_pred, val_returns),
                  source_commit=RELEASE, device=device)
    write_json(root/"result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root.resolve())
