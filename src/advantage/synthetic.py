"""Finite-horizon stochastic robot-progress toy MDP, with exact DP evaluation.

The task horizon is a parameter rather than a module-wide constant, so the
same code can generate and evaluate several horizons. Defaults reproduce the
original 24-step MDP exactly.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .schema import Episode, split_families
from .targets import discounted_returns, n_step_targets, n_step_scores

HORIZON, STATES, POLICY_ADVANCE = 24, 5, 0.6


def outcomes(state, action, states=STATES, regime="constant"):
    if regime == "constant":
        fail, advance = ((0.02, 0.45) if action == 0 else (0.08, 0.75))
    elif regime == "state_dependent":
        # Near the goal the cautious action becomes better; at the start the
        # aggressive one is. This makes the optimal action state dependent,
        # which a constant-preference MDP cannot test.
        progress = state / max(states - 1, 1)
        if action == 0:
            fail, advance = 0.08, 0.45 + 0.30 * progress
        else:
            fail, advance = 0.02 + 0.24 * progress, 0.75
    else:
        raise ValueError(f"unknown regime: {regime}")
    new_state = state + 1
    return [(fail, states, 0., True),
            ((1-fail)*advance, new_state, float(new_state == states), new_state == states),
            ((1-fail)*(1-advance), state, 0., False)]


def exact_values(gamma, horizon=HORIZON, states=STATES, policy_advance=POLICY_ADVANCE,
                 regime="constant"):
    """Evaluation-only oracle; not passed to any training target."""
    if horizon < 1 or states < 1:
        raise ValueError("horizon and states must be positive")
    if not 0 <= policy_advance <= 1:
        raise ValueError("policy_advance must be a probability")
    value = np.zeros((horizon+1, states+1))
    q = np.zeros((horizon, states, 2))
    for t in range(horizon-1, -1, -1):
        for s in range(states):
            for a in range(2):
                q[t, s, a] = sum(p*(r + (0 if done else gamma*value[t+1, nxt]))
                                 for p, nxt, r, done in outcomes(s, a, states, regime))
            value[t, s] = (1-policy_advance)*q[t, s, 0] + policy_advance*q[t, s, 1]
    return value, q


def generate(count=450, seed=0, horizon=HORIZON, states=STATES, policy_advance=POLICY_ADVANCE,
             regime="constant"):
    if horizon < 1 or states < 1:
        raise ValueError("horizon and states must be positive")
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        state = 0
        obs, nxt, actions, rewards = [], [], [], []
        for t in range(horizon):
            action = int(rng.random() < policy_advance)
            choices = outcomes(state, action, states, regime)
            _, next_state, reward, done = choices[rng.choice(3, p=[c[0] for c in choices])]
            obs.append([state]); nxt.append([next_state]); actions.append([action]); rewards.append(reward)
            state = next_state
            if done:
                break
        term = np.zeros(len(rewards), bool); term[-1] = True
        result.append(Episode(f"toy-{seed}-{i}", f"toy-{seed}-{i}", "progress-v1",
            "fixed-random-policy-v1", np.array(obs), np.array(nxt), np.array(actions),
            np.array(rewards), term, np.zeros(len(rewards), bool),
            horizon-np.arange(len(rewards)),
            {"schema_version": 1, "feature_version": "progress-and-time-v1", "seed": seed,
             "horizon": horizon, "states": states, "policy_advance": policy_advance,
             "regime": regime, "success": bool(rewards[-1]), "synthetic": True}))
    return result


def _indices(ep, next_obs=False, horizon=HORIZON):
    time = horizon-ep.remaining_steps + int(next_obs)
    state = (ep.next_obs if next_obs else ep.obs)[:, 0].astype(int)
    return time, state


def _predict(table, ep, nxt=False, horizon=HORIZON):
    return table[_indices(ep, nxt, horizon)]


def branch(state, remaining, action, seed, states=STATES, policy_advance=POLICY_ADVANCE,
           regime="constant"):
    """Force one candidate action, then follow the fixed reference policy.

    Shared by every study so that anchor rollouts use one definition.
    ``remaining`` counts primitive steps left from ``state``; the returned
    episode starts at that state, so its own time index is zero.
    """
    rng = np.random.default_rng(seed)
    obs = []; nxt = []; actions = []; rewards = []
    for i in range(remaining):
        a = action if i == 0 else int(rng.random() < policy_advance)
        options = outcomes(state, a, states, regime)
        _, new, r, done = options[rng.choice(3, p=[o[0] for o in options])]
        obs.append([state]); nxt.append([new]); actions.append([a]); rewards.append(r)
        state = new
        if done:
            break
    term = np.zeros(len(rewards), bool); term[-1] = True
    return Episode(f"branch-{seed}-{action}", f"anchor-{seed}", "progress-v1", "fixed-random-policy-v1",
                   np.array(obs), np.array(nxt), np.array(actions), np.array(rewards), term,
                   np.zeros(len(rewards), bool), remaining - np.arange(len(rewards)),
                   {"synthetic": True, "forced_first_action": True})


def anchor_grid(states=STATES, fractions=(0.1, 0.3, 0.5, 0.7, 0.9), horizon=HORIZON):
    """State/remaining-time anchors that match across horizons.

    A fraction f maps to ``remaining = round(f * horizon)``, so the same
    (state, fraction) pair probes the same normalized time-to-go for every
    horizon. These anchors are synthetic probe states, not sampled frames.
    """
    if not 0 < min(fractions) or max(fractions) > 1:
        raise ValueError("fractions must lie in (0, 1]")
    anchors = []
    for state in range(states):
        seen = set()
        for fraction in fractions:
            remaining = int(round(float(fraction) * horizon))
            remaining = max(1, min(horizon, remaining))
            if remaining in seen:
                continue
            seen.add(remaining)
            anchors.append(dict(state=state, remaining=remaining, fraction=float(fraction)))
    return anchors


def experiment(output, seeds=(0, 1, 2), count=450, epochs=120, gamma=0.99, horizon=HORIZON,
               regime="constant"):
    if count < 30 or epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("use at least 30 episodes, positive epochs and unique seeds")
    if not 0 <= gamma <= 1:
        raise ValueError("invalid gamma")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory for each experiment")
    output.mkdir(parents=True, exist_ok=True)
    oracle, q = exact_values(gamma, horizon, regime=regime)
    curves, rows, scores, manifests = [], [], [], []
    for seed in seeds:
        episodes = generate(count, seed, horizon, regime=regime)
        split = split_families(episodes, seed)
        train = [e for e in episodes if e.episode_id in split["train"]]
        test = [e for e in episodes if e.episode_id in split["test"]]
        train_ids = np.concatenate([np.ravel_multi_index(_indices(e, horizon=horizon), oracle.shape) for e in train])
        counts = np.bincount(train_ids, minlength=oracle.size)
        visited = counts > 0
        test_ids = np.concatenate([np.ravel_multi_index(_indices(e, horizon=horizon), oracle.shape) for e in test])
        truth = oracle.ravel()[test_ids]
        def aggregate(targets):
            sums = np.bincount(train_ids, weights=np.concatenate(targets), minlength=oracle.size)
            return np.divide(sums, counts, out=np.zeros_like(sums), where=visited).reshape(oracle.shape)
        # Shared MC-fitted V for the separate scoring-span comparison.
        shared = aggregate([discounted_returns(e, gamma) for e in train])
        for n in (1, 5, 10, 20):
            table = np.zeros_like(oracle)
            for epoch in range(1, epochs+1):
                target = aggregate([n_step_targets(e, _predict(table, e, True, horizon), n, gamma) for e in train])
                table += 0.2*(target-table)
                curves.append({"seed": seed, "n": n, "epoch": epoch,
                               "test_v_rmse": float(np.sqrt(np.mean((table.ravel()[test_ids]-truth)**2)))})
            predicted_score, true_score = [], []
            for ep in test:
                estimated = n_step_scores(ep, _predict(shared, ep, horizon=horizon),
                                          _predict(shared, ep, True, horizon), n, gamma)
                t, s = _indices(ep, horizon=horizon)
                exact_a = q[t, s, ep.action[:, 0].astype(int)]-oracle[t, s]
                predicted_score.extend(estimated); true_score.extend(exact_a)
                for frame, (pred, true) in enumerate(zip(estimated, exact_a)):
                    scores.append({"episode_id": ep.episode_id, "t": frame, "method": f"n{n}",
                                   "seed": seed, "score": float(pred), "oracle_advantage": float(true)})
            rows.append({"seed": seed, "n": n, "test_v_rmse": curves[-1]["test_v_rmse"],
                         "shared_v_score_rmse": float(np.sqrt(np.mean((np.array(predicted_score)-true_score)**2))),
                         "test_state_coverage": float(visited[test_ids].mean()),
                         "train_episodes": len(train), "test_episodes": len(test),
                         "success_rate": float(np.mean([e.metadata["success"] for e in episodes]))})
            np.savez_compressed(output/f"value_seed{seed}_n{n}.npz", learned=table, oracle=oracle)
        manifests.append({"seed": seed, "split": split})
        # Inspectable example trajectories; synthetic data are not uploaded in bulk.
        for ep in episodes[:3]:
            ep.save(output/f"sample_{ep.episode_id}.npz")
    for name, data in (("metrics.csv", rows), ("learning_curves.csv", curves), ("scores.csv", scores)):
        with (output/name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    summary = [{"n": n, **{key+suffix: float(fn([r[key] for r in rows if r["n"] == n]))
               for key in ("test_v_rmse", "shared_v_score_rmse")
               for suffix, fn in (("_mean", np.mean), ("_std", np.std))}} for n in (1,5,10,20)]
    report = {"config": {"seeds": list(seeds), "episodes_per_seed": count, "epochs": epochs,
              "gamma": gamma, "horizon": horizon, "regime": regime, "learning_rate": 0.2,
              "learner": "synchronous tabular semi-gradient, full training batch"},
              "summary": summary, "runs": rows,
              "limitations": ["Toy MDP only; no real robot or RLinf performance claim.",
                "Equal epochs and transition data; longer n consumes more target-construction CPU.",
                "Score error includes future rollout randomness; it is not a counterfactual ranking metric.",
                "DP oracle is evaluation-only; validation split reserved and not used to tune settings."]}
    (output/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output/"splits.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for n in (1,5,10,20):
            matrix = np.array([[c["test_v_rmse"] for c in curves if c["n"] == n and c["seed"] == s] for s in seeds])
            axes[0].plot(range(1,epochs+1), matrix.mean(0), label=f"n={n}")
        axes[0].set(xlabel="Training epoch", ylabel="Held-out V RMSE vs exact DP", title="Learning V: shared data and initialization")
        axes[0].legend()
        axes[1].bar([str(s["n"]) for s in summary], [s["shared_v_score_rmse_mean"] for s in summary],
                    yerr=[s["shared_v_score_rmse_std"] for s in summary], capsize=4)
        axes[1].set(xlabel="Scoring lookahead n", ylabel="Score RMSE vs exact advantage", title="Fixed MC-trained V: separate scoring test")
        fig.tight_layout(); fig.savefig(output/"comparison.png", dpi=160); plt.close(fig)
    except ImportError:
        pass
    return report
