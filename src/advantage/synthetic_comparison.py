"""Shared model fitting and branch scoring for synthetic estimator studies."""
import numpy as np

from .qvalue import QPredictor, fit_q, qv_scores
from .targets import n_step_scores
from .value import ValuePredictor, fit_value


def estimator_specs(ns=(1, 5, 10, 20), value_ns=(1, 5)):
    """Every estimator with its information requirements, for honest reporting."""
    specs = [dict(name="mc_q_minus_v", family="q_minus_v", lookahead=1,
                  sees_future_observations=False, stochastic_scoring=False,
                  formula="Q(s,a) - V(s)")]
    for n in ns:
        specs.append(dict(name=f"td_n{n}", family="td", lookahead=n,
                          sees_future_observations=True, stochastic_scoring=True,
                          formula=("r + gamma*V(s') - V(s)" if n == 1
                                   else f"sum_k gamma^k r_k + gamma^{n} V(s_{n}) - V(s)")))
    for n in value_ns:
        specs.append(dict(name=f"value_delta_n{n}", family="value_difference", lookahead=n,
                          sees_future_observations=True, stochastic_scoring=True,
                          formula=f"gamma^{n} * V(s_{n}) - V(s), no reward term"))
    return specs


def value_difference_score(episode, values, next_values, n, gamma):
    """gamma^n * V(s_n) - V(s_0) with no reward term.

    Termination inside the window makes the tail value zero; otherwise the
    value predicted for s_n (aligned with next_obs) is discounted back.
    """
    stop = min(n, len(episode))
    tail = 0.0 if episode.terminated[stop - 1] else float(next_values[stop - 1])
    return gamma ** stop * tail - float(values[0])


def score_once(spec, episode, values, next_values, qv_score, gamma):
    """One estimator's score for the forced first action of a branch."""
    if spec["family"] == "q_minus_v":
        return qv_score
    if spec["family"] == "td":
        return float(n_step_scores(episode, values, next_values, spec["lookahead"], gamma)[0])
    return value_difference_score(episode, values, next_values, spec["lookahead"], gamma)


def comparison_branch_seed(seed, anchor, repetition):
    return int(np.random.SeedSequence([seed, anchor, repetition, 9876]).generate_state(1)[0])


def choose_anchors(test, anchors, seed):
    """One anchor state per held-out episode, frame drawn once and reused."""
    rng = np.random.default_rng(100000 + seed)
    chosen = rng.permutation(len(test))[:anchors]
    result = []
    for episode_index in chosen:
        episode = test[int(episode_index)]
        result.append((episode, int(rng.integers(len(episode)))))
    return result


def fit_comparison_models(train, output, seed, updates, gamma):
    """Train the shared MC value models and reload their frozen checkpoints."""
    import torch

    value = fit_value(train, gamma=gamma, updates=updates, seed=seed, method="mc")
    q = fit_q(train, gamma=gamma, updates=updates, seed=seed)
    value_path = output / f"v_seed{seed}.pt"
    q_path = output / f"q_seed{seed}.pt"
    torch.save(value, value_path)
    torch.save(q, q_path)
    return ValuePredictor(value_path), QPredictor(q_path)


def score_branches(branches, specs, vp, qp, gamma, *, reuse_qv=False):
    """Score each rollout of one candidate using the same predicted values."""
    specs_by_name = {spec["name"]: spec for spec in specs}
    scores = {name: [] for name in specs_by_name}
    qv_score = None
    for episode in branches:
        values = vp.predict(episode)
        next_values = vp.predict(episode, True)
        if qp is not None and (qv_score is None or not reuse_qv):
            qv_score = float(qv_scores(qp, vp, episode)[0])
        for name, spec in specs_by_name.items():
            score = score_once(spec, episode, values, next_values, qv_score, gamma)
            scores[name].append(score)
    return scores
