"""Collect frozen single-task MINERVA rollouts; each completed NPZ is a commit marker."""
import argparse
import hashlib
import json
import os
import random
import traceback
from pathlib import Path

import mujoco
import numpy as np
import torch
from advantage.schema import Episode, file_hash
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import make_policy, make_pre_post_processors

WEIGHTS = Path("/mnt/d/CV/minerva_smoke/inputs/t05_0.54M")
TASK = "put the cream cheese in the bowl"
POLICY = "MINERVA-t05_0.54M-38e7414-temp0.85-ensemble0.01-step1"
HORIZON = 300


def family_split():
    order = np.random.default_rng(12).permutation(50).tolist()
    return {name: sorted(ids) for name, ids in
            zip(("train", "validation", "test"), (order[:35], order[35:40], order[40:]))}


def save_snapshot(path, base, policy, raw, frame, seed):
    """Before selecting action at frame. Episode.action[:frame] is the replay prefix."""
    sim = base._env.env.sim
    model, data = sim.model._model, sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    physics = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, physics, spec)
    snapshot = dict(frame=frame, policy_seed=seed, mujoco_state=physics,
                    mj_state_spec=int(spec), torch_rng=torch.get_rng_state().numpy(),
                    python_rng=json.dumps(random.getstate()), sim_time=data.time)
    rng = np.random.get_state()
    snapshot.update(np_rng_keys=rng[1], np_rng_meta=json.dumps([rng[0], *rng[2:]]))
    for key, image in raw["pixels"].items():
        snapshot["current_" + key] = image[0]
    # A fixed prefix or these queues reconstructs temporal policy context.
    for key, queue in policy._queues.items():
        if queue:
            snapshot["queue_" + key] = torch.stack(list(queue)).cpu().numpy()
    ensembler = policy._ensembler
    for attr in ("ensembled_actions", "ensembled_actions_count"):
        value = getattr(ensembler, attr)
        if value is not None:
            snapshot[attr] = value.cpu().numpy()
    controller = base._env.robots[0].controller
    for attr in ("goal_pos", "goal_ori", "initial_ee_pos", "initial_ee_ori_mat"):
        value = getattr(controller, attr, None)
        if value is not None:
            snapshot["controller_" + attr] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.npz")
    np.savez_compressed(temp, **snapshot)
    os.replace(temp, path)


class Collector:
    def __init__(self):
        cfg = PreTrainedConfig.from_pretrained(WEIGHTS)
        cfg.pretrained_path = WEIGHTS
        cfg.device = "cpu"
        cfg.n_action_steps = 1
        cfg.temporal_ensemble_coeff = 0.01
        cfg.noise_temperature = 0.85
        env_cfg = LiberoEnv(task="libero_goal", task_ids=[6])
        self.env = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_goal"][6]
        self.base = self.env.envs[0].unwrapped
        assert self.base.task_description == TASK and cfg.task_to_index[TASK] == 13
        assert len(self.base._init_states) >= 50 and self.base._max_episode_steps == HORIZON
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={}).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=WEIGHTS,
            preprocessor_overrides={"device_processor": {"device": "cpu"},
                                    "rename_observations_processor": {"rename_map": {}}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

    def prepare(self, raw):
        obs = preprocess_observation(raw)
        obs["task"] = [TASK]
        obs = self.env_pre(obs)
        state = obs["observation.state"][0].cpu().numpy().copy()
        assert state.shape == (8,)
        return obs, state

    @torch.inference_mode()
    def collect(self, root, init, rep, split):
        name = f"init{init:02d}_rep{rep}"
        dest = root / "episodes" / (name + ".npz")
        if dest.exists():
            Episode.load(dest)
            return
        env_seed, policy_seed = 11000 + init, 21000 + 2 * init + rep
        self.base.init_state_id = init
        random.seed(env_seed)
        np.random.seed(env_seed)
        self.policy.reset()
        raw, _ = self.env.reset(seed=[env_seed], options={NEW_ROLLOUT_OPTION: True})
        assert self.base.init_state_id == init + 1
        torch.manual_seed(policy_seed)
        initial = np.ascontiguousarray(self.base._init_states[init])
        metadata = dict(init_id=init, env_seed=env_seed, policy_seed=policy_seed, split=split,
                        init_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                        horizon=HORIZON, snapshot_frames=[])
        obs, state = self.prepare(raw)
        states, actions, rewards = [state], [], []
        for t in range(HORIZON):
            if split == "test" and t % 40 == 0:
                save_snapshot(root / "snapshots" / name / f"{t:03d}.npz",
                              self.base, self.policy, raw, t, policy_seed)
                metadata["snapshot_frames"].append(t)
            action = self.policy.select_action(self.pre(obs))
            command = self.env_post({"action": self.post(action)})["action"].cpu().numpy()
            assert command.shape == (1, 7) and np.isfinite(command).all()
            actions.append(command[0].copy())
            raw, reward, terminated, truncated, info = self.env.step(command)
            success = bool(info["is_success"][0])
            assert not truncated[0] and bool(terminated[0]) == success
            assert float(reward[0]) == float(success)
            rewards.append(float(success))
            obs, state = self.prepare(raw)
            states.append(state)
            if success:
                break
        metadata["end_reason"] = "success" if success else "time_limit"
        size = len(rewards)
        terminal = np.zeros(size, dtype=bool)
        terminal[-1] = True
        episode = Episode(name, f"init{init:02d}", "libero_goal_6", POLICY, np.array(states[:-1]),
                          np.array(states[1:]), np.array(actions), np.array(rewards, np.float32),
                          terminal, np.zeros(size, dtype=bool), HORIZON - np.arange(size), metadata)
        temp = dest.with_suffix(".partial.npz")
        episode.save(temp)
        os.replace(temp, dest)
        print(f"{name} {split} {metadata['end_reason']} {size} steps", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("worker", type=int)
    parser.add_argument("workers", type=int)
    args = parser.parse_args()
    collector = Collector()
    try:
        for split, ids in family_split().items():
            for init in ids:
                if init % args.workers != args.worker:
                    continue
                for rep in range(2):
                    try:
                        collector.collect(args.root, init, rep, split)
                    except Exception:
                        crash = args.root / f"crashes_worker{args.worker}.jsonl"
                        with crash.open("a") as f:
                            print(json.dumps(dict(init=init, rep=rep, error=traceback.format_exc())), file=f)
                        raise
    finally:
        collector.env.close()


if __name__ == "__main__":
    main()
