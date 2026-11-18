"""Advantage labels for supervised representations.

The toy MDP has an exact dynamic-programming oracle, so every transition gets
a true advantage. Binary labels deliberately keep an ``ignore`` value for
pairs inside the minimum gap: a sign label there would be noise, not truth.
"""
import numpy as np

IGNORE = -1


def episode_time(episode, horizon):
    """Absolute time index of each transition in an episode."""
    return int(horizon) - np.asarray(episode.remaining_steps, dtype=int)


def build_transitions(episodes, oracle_q, oracle_v, horizon):
    """Flatten episodes into the arrays every representation learns from."""
    obs, remaining, action, advantage, episode_index, time = [], [], [], [], [], []
    for index, episode in enumerate(episodes):
        t = episode_time(episode, horizon)
        state = episode.obs[:, 0].astype(int)
        act = episode.action[:, 0].astype(int)
        obs.append(np.asarray(episode.obs, dtype=np.float64))
        remaining.append(np.asarray(episode.remaining_steps, dtype=np.float64))
        action.append(np.asarray(episode.action, dtype=np.float64))
        advantage.append(oracle_q[t, state, act] - oracle_v[t, state])
        episode_index.append(np.full(len(episode), index, dtype=int))
        time.append(t)
    if not obs:
        raise ValueError("at least one episode required")
    return dict(obs=np.concatenate(obs), remaining=np.concatenate(remaining),
                action=np.concatenate(action), advantage=np.concatenate(advantage),
                episode=np.concatenate(episode_index), time=np.concatenate(time),
                state_index=np.concatenate([e.obs[:, 0].astype(int) for e in episodes]))


def binary_labels(advantage, min_gap=0.005):
    """1 above +min_gap, 0 below -min_gap, IGNORE inside the band."""
    advantage = np.asarray(advantage, dtype=np.float64)
    labels = np.full(advantage.shape, IGNORE, dtype=np.int8)
    labels[advantage > min_gap] = 1
    labels[advantage < -min_gap] = 0
    return labels


def all_cells(states, horizon):
    """Every (state, remaining) cell the toy MDP can visit."""
    return [dict(state=state, remaining=remaining)
            for state in range(states) for remaining in range(1, horizon + 1)]


def anchor_truth(anchors, oracle_q, oracle_v, horizon, n_actions=2):
    """Reference Q, V and advantage for a list of (state, remaining) anchors."""
    q = np.zeros((len(anchors), n_actions))
    v = np.zeros(len(anchors))
    for index, anchor in enumerate(anchors):
        t = int(horizon) - int(anchor["remaining"])
        q[index] = oracle_q[t, int(anchor["state"]), :n_actions]
        v[index] = oracle_v[t, int(anchor["state"])]
    return q, v, q - v[:, None]


def transition_pairs(transition, min_gap=0.005):
    """Same-cell action pairs observed in the training transitions.

    Pairing across cells would teach the model which state is easier instead
    of which action is better, so grouping is by (state, remaining).
    """
    groups = {}
    for index, (state, remaining) in enumerate(zip(transition["state_index"], transition["remaining"])):
        groups.setdefault((int(state), int(remaining)), []).append(index)
    pairs = []
    for indices in groups.values():
        for first in range(len(indices) - 1):
            for second in range(first + 1, len(indices)):
                i, j = indices[first], indices[second]
                if transition["action"][i, 0] == transition["action"][j, 0]:
                    continue
                gap = transition["advantage"][j] - transition["advantage"][i]
                if abs(gap) <= min_gap:
                    continue
                pairs.append(dict(i=int(i), j=int(j), label=1 if gap > 0 else 0))
    return pairs


def split_cells(anchors, holdout=0.25, seed=0):
    """Hold out whole (state, remaining) cells, not just transitions.

    States repeat across episodes, so an episode-level split measures
    interpolation. Holding out cells is what makes an unseen-region claim.
    """
    if not 0 <= holdout < 1:
        raise ValueError("holdout fraction must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(anchors))
    cut = int(round(len(anchors) * holdout))
    held = set(order[:cut].tolist())
    seen = [anchor for index, anchor in enumerate(anchors) if index not in held]
    unseen = [anchor for index, anchor in enumerate(anchors) if index in held]
    return seen, unseen
