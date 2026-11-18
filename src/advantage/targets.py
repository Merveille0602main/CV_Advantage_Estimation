"""Targets for learning V; scores subtract V. Each call processes ONE episode."""
import numpy as np
from .schema import Episode


def _inputs(ep, next_values, gamma):
    ep.validate()
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and between zero and one")
    v = np.asarray(next_values, dtype=np.float64)
    if v.shape != (len(ep),) or not np.isfinite(v).all():
        raise ValueError("next_values must be a finite vector aligned with next_obs")
    return v


def n_step_targets(ep: Episode, next_values, n=1, gamma=0.99):
    """Shorten at file end; bootstrap at truncation, never at true terminal.

    next_values[j] predicts ep.next_obs[j], including its remaining time.
    Finite-task horizon exhaustion must be marked terminated by the adapter.
    """
    v = _inputs(ep, next_values, gamma)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    result = np.zeros(len(ep), dtype=np.float64)
    for t in range(len(ep)):
        stop = min(t + n, len(ep))
        weight = 1.0
        for j in range(t, stop):
            result[t] += weight * ep.reward[j]
            weight *= gamma
        if not ep.terminated[stop-1]:
            result[t] += weight * v[stop-1]
    return result


def td_targets(ep, next_values, gamma=0.99):
    return n_step_targets(ep, next_values, n=1, gamma=gamma)


def n_step_scores(ep, values, next_values, n=1, gamma=0.99):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(ep),) or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector aligned with obs")
    return n_step_targets(ep, next_values, n, gamma) - values


def td_scores(ep, values, next_values, gamma=0.99):
    return n_step_scores(ep, values, next_values, 1, gamma)


def lambda_targets(ep, next_values, gamma=0.99, lambda_=0.9):
    """TD(lambda) forward return, without eligibility traces.

    G_t = r_t + gamma * ((1-lambda) * V(s_{t+1}) + lambda * G_{t+1}), and a
    terminal transition does not bootstrap. ``next_values[t]`` predicts
    ``ep.next_obs[t]``, exactly like ``n_step_targets``.
    """
    v = _inputs(ep, next_values, gamma)
    if not np.isfinite(lambda_) or not 0 <= lambda_ <= 1:
        raise ValueError("lambda must be finite and between zero and one")
    result = np.empty(len(ep), dtype=np.float64)
    tail = 0.0
    for t in range(len(ep) - 1, -1, -1):
        if ep.terminated[t]:
            tail = float(ep.reward[t])
        else:
            tail = float(ep.reward[t]) + gamma * ((1 - lambda_) * v[t] + lambda_ * tail)
        result[t] = tail
    return result


def lambda_scores(ep, values, next_values, gamma=0.99, lambda_=0.9):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(ep),) or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector aligned with obs")
    return lambda_targets(ep, next_values, gamma, lambda_) - values


def discounted_returns(ep, gamma=0.99, bootstrap_value=None):
    """Full returns. Incomplete episodes require an explicit bootstrap estimate."""
    if ep.truncated[-1] and bootstrap_value is None:
        raise ValueError("truncated episode has unknown full return; supply bootstrap_value explicitly")
    tail = 0.0 if ep.terminated[-1] else float(bootstrap_value)
    v = np.zeros(len(ep), dtype=float)
    v[-1] = tail
    _inputs(ep, v, gamma)
    result = np.empty(len(ep), dtype=float)
    for j in range(len(ep)-1, -1, -1):
        tail = ep.reward[j] + gamma * (0.0 if ep.terminated[j] else tail)
        result[j] = tail
    return result
