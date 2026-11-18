"""Option-2 study: temporal horizon, discount factor and value-target effects.

Two questions are kept separate:

1. Does the advantage estimate change when the task horizon or the discount
   factor changes? Each (gamma, horizon) cell recomputes its own exact
   dynamic-programming oracle and probes the same normalized anchors, so the
   comparison never uses one setting's labels for another setting.
2. Does the value target change the resulting advantage estimate? The
   advantage construction is held fixed while V is trained with MC returns,
   n-step bootstrapping, TD(lambda) returns, or an oracle "teacher" target
   that removes bootstrapping error entirely.

Anchors are synthetic (state, fraction-of-horizon) probes, not sampled
frames. That makes the same state/remaining-time pairs available for every
horizon, which episode sampling cannot guarantee.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .schema import split_families
from .synthetic import (generate, exact_values, branch, anchor_grid,
                        STATES, POLICY_ADVANCE)
from .value import fit_value, ValuePredictor
from .qvalue import fit_q, QPredictor, qv_scores
from .targets import n_step_scores
from . import discrimination

TARGETS = ("mc", "nstep1", "nstep5", "lambda0.9")
SCORING = ("td1",)


def parse_target(spec):
    """Map a value-target label to fit_value arguments."""
    if spec == "mc":
        return dict(method="mc", n=1, lambda_=0.9)
    if spec == "teacher":
        return dict(method="teacher", n=1, lambda_=0.9)
    if spec.startswith("nstep"):
        try:
            n = int(spec[len("nstep"):])
        except ValueError:
            raise ValueError(f"nstep target needs an integer lookahead: {spec}")
        if n < 1:
            raise ValueError("nstep lookahead must be positive")
        return dict(method="nstep", n=n, lambda_=0.9)
    if spec.startswith("lambda"):
        try:
            value = float(spec[len("lambda"):])
        except ValueError:
            raise ValueError(f"lambda target needs a numeric trace decay: {spec}")
        if not 0 <= value <= 1:
            raise ValueError("lambda must be between zero and one")
        return dict(method="lambda", n=1, lambda_=value)
    raise ValueError(f"unknown value target: {spec}")


def parse_scoring(spec):
    """Map a scoring label to the advantage construction it computes."""
    if spec == "qv":
        return dict(name=spec, kind="q_minus_v", lookahead=1)
    if spec.startswith("td"):
        try:
            n = int(spec[len("td"):])
        except ValueError:
            raise ValueError(f"td scoring needs an integer lookahead: {spec}")
        if n < 1:
            raise ValueError("td lookahead must be positive")
        return dict(name=spec, kind="n_step", lookahead=n)
    raise ValueError(f"unknown scoring rule: {spec}")


def _tag(value):
    return f"{float(value):.2f}".replace(".", "p")


def _branch_seed(gamma, horizon, seed, anchor, repetition):
    parts = [int(round(float(gamma) * 1000)), int(horizon), int(seed), int(anchor), int(repetition)]
    return int(np.random.SeedSequence(parts).generate_state(1)[0])


def _write_csv(path, rows, columns=None):
    if not rows:
        if not columns:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(columns)).writeheader()
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def teacher_source(oracle_values, horizon):
    """Per-step exact V, used only by the explicit teacher diagnostic."""
    def source(episode):
        return oracle_values[horizon - episode.remaining_steps, episode.obs[:, 0].astype(int)]
    return source


def experiment(output, gammas=(0.9, 0.95), horizons=(24, 48), targets=TARGETS,
               scoring=SCORING, seeds=(0, 1, 2), episodes=450, updates=1000,
               states=STATES, fractions=(0.1, 0.3, 0.5, 0.7, 0.9), repeats=16,
               min_gap=0.005, n_boot=2000, bootstrap_seed=0, lr=3e-4,
               target_refresh=20, hidden=(256, 256), policy_advance=POLICY_ADVANCE,
               regime="constant"):
    import torch
    if not gammas or not horizons or not targets or not scoring or not seeds:
        raise ValueError("at least one gamma, horizon, target, scoring rule and seed required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("unique seeds required")
    if episodes < 30 or updates < 1 or repeats < 1 or n_boot < 1 or target_refresh < 1:
        raise ValueError("positive budgets, at least 30 episodes required")
    if any(not 0 <= float(g) <= 1 for g in gammas):
        raise ValueError("gammas must lie in [0, 1]")
    if any(int(h) < 1 for h in horizons) or int(states) < 1:
        raise ValueError("positive horizon and states required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rules = [parse_scoring(spec) for spec in scoring]
    target_specs = {spec: parse_target(spec) for spec in targets}

    cells, anchor_rows, progress_rows, split_rows = [], [], [], []
    correctness = {}
    for gamma in gammas:
        for horizon in horizons:
            oracle_v, oracle_q = exact_values(gamma, horizon, states, policy_advance, regime)
            anchors = anchor_grid(states, fractions, horizon)
            if not anchors:
                raise ValueError("anchor grid is empty")
            n_actions = oracle_q.shape[2]
            truth = np.zeros((len(anchors), n_actions))
            true_v = np.zeros(len(anchors))
            for index, anchor in enumerate(anchors):
                t = horizon - anchor["remaining"]
                truth[index] = oracle_q[t, anchor["state"], :]
                true_v[index] = oracle_v[t, anchor["state"]]
            advantage_truth = truth - true_v[:, None]

            for seed in seeds:
                data = generate(episodes, seed, horizon, states, policy_advance, regime)
                split = split_families(data, seed)
                train = [e for e in data if e.episode_id in split["train"]]
                success_rate = float(np.mean([e.metadata["success"] for e in data]))
                split_rows.append(dict(gamma=gamma, horizon=horizon, seed=seed, split=split))
                qp = None
                if any(rule["kind"] == "q_minus_v" for rule in rules):
                    qc = fit_q(train, gamma=gamma, updates=updates, seed=seed, lr=lr)
                    qpath = output / f"q_g{_tag(gamma)}_h{horizon}_s{seed}.pt"
                    torch.save(qc, qpath)
                    qp = QPredictor(qpath)

                for target in targets:
                    spec = target_specs[target]
                    arguments = dict(gamma=gamma, updates=updates, seed=seed, lr=lr,
                                     target_refresh=target_refresh, method=spec["method"],
                                     n=spec["n"], lambda_=spec["lambda_"], hidden=hidden)
                    if spec["method"] == "teacher":
                        arguments["target_source"] = teacher_source(oracle_v, horizon)
                    checkpoint = fit_value(train, **arguments)
                    vpath = output / f"v_g{_tag(gamma)}_h{horizon}_{target}_s{seed}.pt"
                    torch.save(checkpoint, vpath)
                    vp = ValuePredictor(vpath)

                    repeats_by_rule = {rule["name"]: np.zeros((len(anchors), n_actions, repeats))
                                       for rule in rules}
                    for ai, anchor in enumerate(anchors):
                        state, remaining = anchor["state"], anchor["remaining"]
                        for action in range(n_actions):
                            learned = None
                            for repetition in range(repeats):
                                sample = branch(state, remaining, action,
                                                _branch_seed(gamma, horizon, seed, ai, repetition),
                                                states, policy_advance, regime)
                                values = vp.predict(sample)
                                next_values = vp.predict(sample, True)
                                if learned is None and qp is not None:
                                    # Q-V depends only on the first state and
                                    # action, so one evaluation per candidate
                                    # is enough; repeated branches would be
                                    # identical draws of the same number.
                                    learned = float(qv_scores(qp, vp, sample)[0])
                                for rule in rules:
                                    if rule["kind"] == "q_minus_v":
                                        score = learned
                                    else:
                                        score = float(n_step_scores(sample, values, next_values,
                                                                    rule["lookahead"], gamma)[0])
                                    repeats_by_rule[rule["name"]][ai, action, repetition] = score

                    for rule in rules:
                        name = rule["name"]
                        repeated = repeats_by_rule[name]
                        mean_scores = repeated.mean(axis=2)
                        anchor_ids = [f"s{a['state']}:f{a['fraction']:.2f}" for a in anchors]
                        records = discrimination.pairwise_records(mean_scores, truth, min_gap, anchor_ids)
                        stats = discrimination.aggregate_metrics(records)
                        stability = discrimination.stability_metrics(repeated, truth, min_gap)
                        error = discrimination.advantage_error_metrics(mean_scores, advantage_truth)
                        interval = discrimination.bootstrap_accuracy(mean_scores, truth, min_gap,
                                                                     n_boot, bootstrap_seed)
                        cells.append(dict(gamma=gamma, horizon=horizon, target=target, scoring=name,
                                          seed=seed, accuracy=stats["accuracy"],
                                          ci_low=interval["ci_low"], ci_high=interval["ci_high"],
                                          pair_coverage=stats["pair_coverage"],
                                          advantage_rmse=error["advantage_rmse"],
                                          reference_spread=error["reference_spread"],
                                          normalized_rmse=error["normalized_rmse"],
                                          calibration_slope=error["calibration_slope"],
                                          rank_flip_rate=stability["rank_flip_rate"],
                                          mean_score_std=stability["mean_score_std"],
                                          success_rate=success_rate, anchors=len(anchors)))
                        groups = {ai: f"{anchor['fraction']:.2f}" for ai, anchor in enumerate(anchors)}
                        for row in discrimination.accuracy_by_group(records, groups):
                            progress_rows.append(dict(gamma=gamma, horizon=horizon, target=target,
                                                      scoring=name, seed=seed, fraction=row["group"],
                                                      pairs=row["pairs"], accuracy=row["accuracy"]))
                        for record in records:
                            anchor = anchors[record["anchor_index"]]
                            anchor_rows.append(dict(gamma=gamma, horizon=horizon, target=target,
                                                    scoring=name, seed=seed, state=anchor["state"],
                                                    fraction=anchor["fraction"], remaining=anchor["remaining"],
                                                    pred_gap=record["pred_gap"], true_gap=record["true_gap"],
                                                    distinguishable=record["distinguishable"],
                                                    correct=record["correct"]))
                        correctness[(gamma, horizon, target, name, seed)] = discrimination.correctness_map(records)

    summary = []
    cell_keys = sorted({(c["gamma"], c["horizon"], c["target"], c["scoring"]) for c in cells})
    for gamma, horizon, target, name in cell_keys:
        subset = [c for c in cells if (c["gamma"], c["horizon"], c["target"], c["scoring"]) ==
                  (gamma, horizon, target, name)]
        accuracies = [c["accuracy"] for c in subset if c["accuracy"] is not None]
        normalised = [c["normalized_rmse"] for c in subset if c["normalized_rmse"] is not None]
        slopes = [c["calibration_slope"] for c in subset if c["calibration_slope"] is not None]
        summary.append(dict(gamma=gamma, horizon=horizon, target=target, scoring=name,
                            accuracy_mean=float(np.mean(accuracies)) if accuracies else None,
                            accuracy_std=float(np.std(accuracies)) if accuracies else None,
                            normalized_rmse_mean=float(np.mean(normalised)) if normalised else None,
                            calibration_slope_mean=float(np.mean(slopes)) if slopes else None,
                            rank_flip_rate_mean=float(np.mean([c["rank_flip_rate"] for c in subset])),
                            success_rate_mean=float(np.mean([c["success_rate"] for c in subset])),
                            seeds=len(subset)))

    def marginal(field):
        values = sorted({row[field] for row in summary})
        rows = []
        for value in values:
            subset = [row for row in summary if row[field] == value]
            accuracies = [row["accuracy_mean"] for row in subset if row["accuracy_mean"] is not None]
            normalised = [row["normalized_rmse_mean"] for row in subset
                          if row["normalized_rmse_mean"] is not None]
            rows.append(dict(value=value, cells=len(subset),
                             accuracy_mean=float(np.mean(accuracies)) if accuracies else None,
                             normalized_rmse_mean=float(np.mean(normalised)) if normalised else None))
        return rows

    def consistency(axis, fixed_fields, varying):
        grouped = {}
        for key in correctness:
            record = dict(zip(("gamma", "horizon", "target", "scoring", "seed"), key))
            grouped.setdefault(tuple(record[field] for field in fixed_fields), {})[record[axis]] = correctness[key]
        rows = []
        for fixed, by_value in grouped.items():
            available = [value for value in varying if value in by_value]
            for position, first in enumerate(available):
                for second in available[position + 1:]:
                    left, right = by_value[first], by_value[second]
                    shared = sorted(set(left) & set(right))
                    if not shared:
                        continue
                    agreement = float(np.mean([1.0 if left[key] == right[key] else 0.0 for key in shared]))
                    rows.append(dict(axis=axis,
                                     fixed="|".join(f"{name}={value}" for name, value in zip(fixed_fields, fixed)),
                                     setting_a=first, setting_b=second,
                                     shared_anchors=len(shared), agreement=agreement))
        return rows

    consistency_rows = []
    consistency_rows += consistency("gamma", ("horizon", "target", "scoring", "seed"), list(gammas))
    consistency_rows += consistency("horizon", ("gamma", "target", "scoring", "seed"), list(horizons))
    consistency_rows += consistency("target", ("gamma", "horizon", "scoring", "seed"), list(targets))

    report = dict(
        config=dict(gammas=list(gammas), horizons=list(horizons), targets=list(targets),
                    scoring=list(scoring), seeds=list(seeds), episodes_per_seed=episodes,
                    updates=updates, states=states, fractions=list(fractions),
                    repeats=repeats,
                    min_gap=min_gap, bootstrap_samples=n_boot, bootstrap_seed=bootstrap_seed,
                    learning_rate=lr, target_refresh=target_refresh,
                    hidden=[int(units) for units in hidden],
                    policy_advance=policy_advance,
                    regime=regime,
                    anchors="state x fraction-of-horizon grid, shared across settings",
                    candidates="all actions of the toy MDP"),
        cells=summary, marginals=dict(gamma=marginal("gamma"), horizon=marginal("horizon"),
                                      target=marginal("target")),
        consistency=consistency_rows,
        limitations=[
            "Toy finite-horizon MDP only; no LIBERO or RLinf claim.",
            "Each (gamma, horizon) cell uses its own exact DP oracle; raw RMSE is not comparable across cells, normalized RMSE is.",
            "Changing gamma changes the target itself, so only within-cell comparisons and rank agreement are meaningful.",
            "Teacher targets are a diagnostic that bypasses bootstrapping, not a deployable estimator.",
            "Q-V is retrained per cell; its error mixes V and Q learning.",
            "Anchor rollouts are synthetic probes, not sampled test trajectories.",
        ])
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(split_rows, indent=2), encoding="utf-8")
    for name, rows in (("cells.csv", cells), ("anchors.csv", anchor_rows),
                       ("progress.csv", progress_rows)):
        _write_csv(output / name, rows)
    _write_csv(output / "consistency.csv", consistency_rows,
               columns=("axis", "fixed", "setting_a", "setting_b", "shared_anchors", "agreement"))
    plot(output, summary)
    return report


def plot(output, summary):
    """Optional figure; absence of matplotlib must not fail the experiment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [f"{row['target']}\ng={row['gamma']},H={row['horizon']}" for row in summary]
        values = [row["accuracy_mean"] or 0.0 for row in summary]
        figure, axis = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4))
        axis.bar(range(len(labels)), values)
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        axis.set(ylabel="Paired ranking accuracy", title="Value target x (gamma, horizon)")
        figure.tight_layout()
        figure.savefig(output / "accuracy_by_setting.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
