"""Fixed MINERVA collector. Frames are commands sent to the LIBERO environment."""
import argparse
import hashlib
import random

import numpy as np
import torch
from advantage.schema import Episode
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import make_policy, make_pre_post_processors
from study import POLICY, TASKS, WEIGHTS, base_dir, eid, save_episode, split


class Simulation:
    def __init__(self, task):
        self.task = task
        suite, index, self.language, self.horizon = TASKS[task]
        cfg = PreTrainedConfig.from_pretrained(WEIGHTS)
        cfg.pretrained_path = WEIGHTS
        cfg.device = "cpu"
        cfg.n_action_steps = 1
        cfg.temporal_ensemble_coeff = 0.01
        cfg.noise_temperature = 0.85
        env_cfg = LiberoEnv(task=suite, task_ids=[index])
        self.env = make_env(env_cfg, n_envs=1, use_async_envs=False)[suite][index]
        self.base = self.env.envs[0].unwrapped
        assert self.base.task_description == self.language and cfg.task_to_index[self.language] == task
        assert len(self.base._init_states) >= 50 and self.base._max_episode_steps == self.horizon
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={}).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=WEIGHTS,
            preprocessor_overrides={"device_processor":{"device":"cpu"}, "rename_observations_processor":{"rename_map":{}}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        self.task_id = f"{suite}_{index}"

    def prepare(self, raw):
        obs = preprocess_observation(raw)
        obs["task"] = [self.language]
        obs = self.env_pre(obs)
        state = obs["observation.state"][0].cpu().numpy().copy()
        assert state.shape == (8,)
        return obs, state

    def start(self, init, rep):
        env_seed, policy_seed = 11000+init, 21000+2*init+rep
        self.base.init_state_id = init
        random.seed(env_seed)
        np.random.seed(env_seed)
        self.policy.reset()
        raw, _ = self.env.reset(seed=[env_seed], options={NEW_ROLLOUT_OPTION:True})
        assert self.base.init_state_id == init+1
        torch.manual_seed(policy_seed)
        return raw, self.prepare(raw)

    def action(self, obs):
        normalized = self.policy.select_action(self.pre(obs))
        return self.command(self.post(normalized))

    def command(self, unnormalized):
        command = self.env_post({"action":unnormalized})["action"].cpu().numpy()[0].copy()
        assert command.shape == (7,) and np.isfinite(command).all()
        return command

    def step(self, command):
        raw, reward, term, trunc, info = self.env.step(command[None])
        success = bool(info["is_success"][0])
        assert success == bool(term[0]) and not trunc[0] and float(reward[0]) == float(success)
        return raw, self.prepare(raw), success

    def rollout(self, obs, state, frame, cap, forced=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        states, actions, rewards = [state], [], []
        for t in range(min(cap, self.horizon-frame)):
            command = self.action(obs)
            if t == 0 and forced is not None:
                command = forced.copy()
            _, (obs, state), success = self.step(command)
            states.append(state)
            actions.append(command)
            rewards.append(float(success))
            if success:
                break
        return np.asarray(states), np.asarray(actions), np.array(rewards, dtype=np.float32)

    def episode(self, history, init, name, frame, meta):
        states, actions, rewards = history
        remaining = self.horizon-frame-np.arange(len(actions))
        term = np.zeros(len(actions), bool)
        trunc = term.copy()
        term[-1] = bool(rewards[-1]) or remaining[-1] == 1
        trunc[-1] = not term[-1]
        return Episode(name, f"init{init:02d}", self.task_id, POLICY, states[:-1], states[1:],
                       actions, rewards, term, trunc, remaining, meta)


@torch.inference_mode()
def collect(task, worker, workers):
    assert task != 13
    sim = Simulation(task)
    try:
        for part, ids in split().items():
            for init in ids:
                if init % workers != worker:
                    continue
                for rep in range(2):
                    name = eid(init, rep)
                    dest = base_dir(task)/"episodes"/(name+".npz")
                    if dest.exists():
                        Episode.load(dest)
                        continue
                    _, (obs, state) = sim.start(init, rep)
                    history = sim.rollout(obs, state, 0, sim.horizon)
                    initial = np.ascontiguousarray(sim.base._init_states[init])
                    meta = dict(init_id=init, split=part, env_seed=11000+init, policy_seed=21000+2*init+rep,
                                init_sha256=hashlib.sha256(initial.tobytes()).hexdigest(), horizon=sim.horizon)
                    ep = sim.episode(history, init, name, 0, meta)
                    assert ep.terminated[-1]
                    save_episode(dest, ep)
                    print(task, part, name, len(ep), ep.reward[-1], flush=True)
    finally:
        sim.env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("task", "worker", "workers"):
        p.add_argument(key,type=int)
    a = p.parse_args()
    collect(a.task,a.worker,a.workers)
