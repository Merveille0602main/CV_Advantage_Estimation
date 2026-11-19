"""Independent score and reference continuations from held-out MINERVA histories."""
import argparse
import json
import os
import random
from pathlib import Path

import mujoco
import numpy as np
import torch
from advantage.schema import Episode, file_hash
from collect import Collector, HORIZON, NEW_ROLLOUT_OPTION, WEIGHTS

SOURCE = Path("/mnt/d/CV/training_runs/minerva_goal13/20261119_run01")
N_REF, N_SCORE = 8, 2
NAMES = ["recorded", "resampled", "perturbed"]


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    os.replace(temp, path)


def reset_history(c, ep, frame, z):
    """Replay logged commands to reconstruct controller, wrapper and observation caches."""
    seed = ep.metadata["env_seed"]
    random.seed(seed)
    np.random.seed(seed)
    c.base.init_state_id = ep.metadata["init_id"]
    c.policy.reset()
    raw, _ = c.env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    for command in ep.action[:frame]:
        raw, _, term, trunc, _ = c.env.step(command[None])
        assert not term[0] and not trunc[0]
    obs, state = c.prepare(raw)
    np.testing.assert_allclose(state, ep.obs[frame], atol=1e-7, rtol=0)
    sim = c.base._env.env.sim
    physics = np.empty_like(z["mujoco_state"])
    mujoco.mj_getState(sim.model._model, sim.data._data, physics, int(z["mj_state_spec"]))
    np.testing.assert_allclose(physics, z["mujoco_state"], atol=1e-8, rtol=0)
    for key, pixels in raw["pixels"].items():
        np.testing.assert_array_equal(pixels[0], z["current_"+key])
    for key, queue in c.policy._queues.items():
        if "queue_"+key in z:
            queue.extend(torch.from_numpy(arr.copy()) for arr in z["queue_"+key])
    for key in ("ensembled_actions", "ensembled_actions_count"):
        if key in z:
            setattr(c.policy._ensembler, key, torch.from_numpy(z[key].copy()))
    rng = json.loads(str(z["python_rng"]))
    random.setstate((rng[0], tuple(rng[1]), rng[2]))
    np_info = json.loads(str(z["np_rng_meta"]))
    np.random.set_state((np_info[0], z["np_rng_keys"], *np_info[1:]))
    torch.set_rng_state(torch.from_numpy(z["torch_rng"].copy()))
    return obs, state


def proposal(c, obs):
    normalized = c.policy.select_action(c.pre(obs))
    return c.env_post({"action": c.post(normalized)})["action"].cpu().numpy()[0].copy()


def run(c, obs, state, ep, frame, seed=None, forced=None, cap=None, reproduce=False):
    if seed is not None:
        torch.manual_seed(seed)
    actions, rewards, next_states = [], [], []
    duration = HORIZON-frame
    limit = min(duration, cap or duration)
    max_state_err = max_action_err = 0.0
    for t in range(limit):
        command = proposal(c, obs)
        if t == 0 and forced is not None:
            command = forced.copy()
        raw, reward, term, trunc, info = c.env.step(command[None])
        success = bool(info["is_success"][0])
        assert bool(term[0]) == success and not trunc[0] and float(reward[0]) == float(success)
        obs, after = c.prepare(raw)
        if reproduce:
            i = frame+t
            max_state_err = max(max_state_err, float(np.max(np.abs(after-ep.next_obs[i]))))
            max_action_err = max(max_action_err, float(np.max(np.abs(command-ep.action[i]))))
            np.testing.assert_allclose(command, ep.action[i], rtol=0, atol=1e-6)
            np.testing.assert_allclose(after, ep.next_obs[i], rtol=0, atol=1e-6)
            assert reward[0] == ep.reward[i]
        actions.append(command)
        rewards.append(float(success))
        next_states.append(after)
        if success:
            break
    steps = len(rewards)
    if reproduce:
        assert frame+steps == len(ep)
    terminal = success or steps == duration
    return dict(state=state.tolist(), actions=np.array(actions).tolist(),
                next_states=np.array(next_states).tolist(), rewards=rewards, terminal=terminal,
                steps=steps, success=success, ret=(0.99**(steps-1) if success else 0.0),
                max_state_err=max_state_err, max_action_err=max_action_err)


def setup():
    manifest = json.loads((SOURCE/"manifest.json").read_text())
    provenance = json.loads((SOURCE/"provenance.json").read_text())
    assert file_hash(SOURCE/"manifest.json") == provenance["manifest_sha256"]
    for name, digest in provenance["weight_files"].items():
        assert file_hash(WEIGHTS/name) == digest
    snaps = json.loads((SOURCE/"snapshot_hashes.json").read_text())
    rows = sorted((r for r in manifest["episodes"] if r["split"]=="test" and
                   r["episode_id"].endswith("_rep0")), key=lambda r: r["init_id"])
    assert len(rows) == len({r["parent_id"] for r in rows}) == 10
    anchors = []
    for i,r in enumerate(rows):
        frame = 40 if i % 2 == 0 else 80
        eid = r["episode_id"]
        ep_path = SOURCE/r["path"]
        snap = f"snapshots/{eid}/{frame:03d}.npz"
        assert file_hash(ep_path) == r["sha256"] and file_hash(SOURCE/snap) == snaps[snap]
        with np.load(SOURCE/snap, allow_pickle=False) as data:
            z = dict(data)
        anchors.append((eid, frame, Episode.load(ep_path), z))
    return anchors


@torch.inference_mode()
def execute(args):
    root = args.output
    root.mkdir(exist_ok=True, parents=True)
    anchors = setup()
    c = Collector()
    try:
        if args.mode=="verify":
            for i,(eid,frame,ep,z) in enumerate(anchors):
                if i % args.workers != args.index:
                    continue
                path = root/f"{eid}_anchor.json"
                if path.exists():
                    continue
                obs,state = reset_history(c,ep,frame,z)
                replay = run(c,obs,state,ep,frame,reproduce=True)
                obs,_ = reset_history(c,ep,frame,z)
                torch.manual_seed(70000+i)
                alternate = proposal(c,obs)
                original = ep.action[frame].copy()
                perturbed = original.copy()
                perturbed[:3] = np.clip(-original[:3], -1, 1)
                perturbed[6] = -1.0 if original[6]>=0 else 1.0
                actions = [original.tolist(), alternate.tolist(), perturbed.tolist()]
                write_json(path, dict(episode=eid,frame=frame,family=ep.parent_id,state=state.tolist(),
                                      candidate_seed=70000+i, candidates=dict(zip(NAMES,actions)),
                                      replay={k:replay[k] for k in ("steps","success","max_state_err","max_action_err")}))
                print("verified",eid,frame,replay["steps"],flush=True)
            return
        # Interleave repeats among workers, including those from longer timeouts.
        for rep in range(N_REF+N_SCORE):
            group = "ref" if rep < N_REF else "score"
            for i,(eid,frame,ep,z) in enumerate(anchors):
                if (i+rep) % args.workers != args.index:
                    continue
                anchor = json.loads((root/f"{eid}_anchor.json").read_text())
                seed = (100000 if group=="ref" else 200000) + 100*i+rep
                # Identical continuation noise within a pair, disjoint streams across purposes.
                path = root/f"{eid}_{group}_{rep:02d}.json"
                if path.exists():
                    continue
                rows = {}
                for name in NAMES:
                    obs,state = reset_history(c,ep,frame,z)
                    r = run(c,obs,state,ep,frame,seed,np.array(anchor["candidates"][name],np.float32),
                            cap=20 if group=="score" else None)
                    rows[name] = (r if group=="score" else
                                  {k:r[k] for k in ("steps","success","ret","terminal")})
                write_json(path, dict(episode=eid,frame=frame,group=group,seed=seed,rows=rows))
                print("finished",eid,group,rep,flush=True)
    finally:
        c.env.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("mode",choices=["verify","branch"])
    parser.add_argument("index",type=int)
    parser.add_argument("workers",type=int)
    execute(parser.parse_args())
