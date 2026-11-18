"""Option-1 study: compare advantage constructions on local action quality.

Compares, on the same anchors and candidate actions:

  mc_q_minus_v       Q(s,a) - V(s), learned from complete reference-policy returns
  td_n1..td_n20      r + gamma*V(s') - V(s) over one or several observed steps
  value_delta_n1/n5  gamma^n * V(s_n) - V(s), no intermediate reward

The last family is the plain temporal value difference some readings mean by
"temporal value difference"; it is kept separate from TD so that the effect of
the reward term stays visible instead of being mixed in silently.

Reference labels come from the toy MDP's exact dynamic programming solution.
They never supervise V, Q, the checkpoint choice, or the scoring rule.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .schema import split_families
from .synthetic import generate, exact_values, branch
from .value import fit_value, ValuePredictor
from .qvalue import fit_q, QPredictor, qv_scores
from .targets import n_step_scores
from . import discrimination


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


def _branch_seed(seed, anchor, repetition, salt=9876):
    return int(np.random.SeedSequence([seed, anchor, repetition, salt]).generate_state(1)[0])


def _choose_anchors(test, anchors, seed):
    """One anchor state per held-out episode, frame drawn once and reused."""
    rng = np.random.default_rng(100000 + seed)
    chosen = rng.permutation(len(test))[:anchors]
    result = []
    for episode_index in chosen:
        episode = test[int(episode_index)]
        frame = int(rng.integers(len(episode)))
        result.append((episode, frame))
    return result


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment(output, seeds=(0, 1, 2), episodes=450, updates=1000, anchors=40, repeats=16,
               min_gap=0.005, n_boot=2000, bootstrap_seed=0, gamma=0.99,
               ns=(1, 5, 10, 20), value_ns=(1, 5)):
    """Run the estimator comparison and write every artifact the report needs."""
    import torch
    if episodes < 30 or updates < 1 or anchors < 1 or repeats < 1 or n_boot < 1:
        raise ValueError("positive budgets, at least 30 episodes required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique seeds required")
    if min_gap < 0 or not 0 <= gamma <= 1:
        raise ValueError("min_gap must be nonnegative and gamma in [0, 1]")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    specs = estimator_specs(tuple(ns), tuple(value_ns))
    methods = [spec["name"] for spec in specs]
    # Oracle arrays exist only for evaluation; no learner below receives them.
    oracle_v, oracle_q = exact_values(gamma)
    candidate_rows, metric_rows, anchor_rows, pair_rows = [], [], [], []
    stability_rows, gap_rows, comparison_rows, split_rows = [], [], [], []
    pooled = {name: dict(pred=[], truth=[], anchor=[]) for name in methods}

    for seed in seeds:
        data = generate(episodes, seed)
        split = split_families(data, seed)
        train = [e for e in data if e.episode_id in split["train"]]
        test = [e for e in data if e.episode_id in split["test"]]
        vc = fit_value(train, gamma=gamma, updates=updates, seed=seed, method="mc")
        qc = fit_q(train, gamma=gamma, updates=updates, seed=seed)
        vp_path, qp_path = output / f"v_seed{seed}.pt", output / f"q_seed{seed}.pt"
        torch.save(vc, vp_path); torch.save(qc, qp_path)
        vp, qp = ValuePredictor(vp_path), QPredictor(qp_path)

        chosen = _choose_anchors(test, anchors, seed)
        n_actions = oracle_q.shape[2]  # action axis of the exact solution
        repeats_by_method = {name: np.zeros((len(chosen), n_actions, repeats)) for name in methods}
        truth = np.zeros((len(chosen), n_actions))
        true_v = np.zeros(len(chosen))
        anchor_ids, episode_ids, frames = [], [], []

        for anchor, (episode, frame) in enumerate(chosen):
            state = int(episode.obs[frame, 0]); remaining = int(episode.remaining_steps[frame])
            # The episode starts at its own horizon, so this recovers the
            # absolute time index without assuming a module-wide constant.
            t = int(episode.remaining_steps[0]) - remaining
            truth[anchor] = oracle_q[t, state, :]
            true_v[anchor] = oracle_v[t, state]
            anchor_ids.append(f"{episode.episode_id}:{frame}")
            episode_ids.append(episode.episode_id); frames.append(frame)
            for action in range(n_actions):
                for repetition in range(repeats):
                    sample = branch(state, remaining, action,
                                    _branch_seed(seed, anchor, repetition))
                    values, next_values = vp.predict(sample), vp.predict(sample, True)
                    qv_score = float(qv_scores(qp, vp, sample)[0])
                    for spec in specs:
                        repeats_by_method[spec["name"]][anchor, action, repetition] = score_once(
                            spec, sample, values, next_values, qv_score, gamma)

        scores = {name: repeats_by_method[name].mean(axis=2) for name in methods}
        advantage_truth = truth - true_v[:, None]

        for name in methods:
            records = discrimination.pairwise_records(scores[name], truth, min_gap, anchor_ids)
            stats = discrimination.aggregate_metrics(records)
            stability = discrimination.stability_metrics(repeats_by_method[name], truth, min_gap)
            majority = discrimination.majority_ordering_accuracy(repeats_by_method[name], truth, min_gap)
            interval = discrimination.bootstrap_accuracy(scores[name], truth, min_gap, n_boot,
                                                         bootstrap_seed + seed)
            metric_rows.append(dict(seed=seed, method=name, **stats, **stability,
                                    majority_accuracy=majority["accuracy"],
                                    ci_low=interval["ci_low"], ci_high=interval["ci_high"],
                                    advantage_rmse=float(np.sqrt(np.mean((scores[name] - advantage_truth) ** 2))),
                                    **discrimination.margin_calibration(records)))
            for row in discrimination.anchor_records(records):
                anchor_rows.append(dict(seed=seed, method=name, anchor_id=row["anchor_id"],
                                        pairs=row["pairs"], valid_pairs=row["valid_pairs"],
                                        accuracy=row["accuracy"], evaluable=row["evaluable"],
                                        true_gap=row["true_gap"], pred_gap=row["pred_gap"]))
            for record in records:
                pair_rows.append(dict(seed=seed, method=name, anchor_id=record["anchor_id"],
                                      candidate_i=record["candidate_i"], candidate_j=record["candidate_j"],
                                      pred_gap=record["pred_gap"], true_gap=record["true_gap"],
                                      distinguishable=record["distinguishable"], correct=record["correct"]))
            for row in discrimination.gap_bin_accuracy(records):
                gap_rows.append(dict(seed=seed, method=name, **row))
            stability_rows.append(dict(seed=seed, method=name, anchors=len(chosen), **stability))
            for anchor, action in np.ndindex(scores[name].shape):
                candidate_rows.append(dict(seed=seed, method=name, anchor_id=anchor_ids[anchor],
                                           episode_id=episode_ids[anchor], t=frames[anchor],
                                           action=int(action), score=float(scores[name][anchor, action]),
                                           score_std=float(repeats_by_method[name][anchor, action].std()),
                                           true_advantage=float(truth[anchor, action] - true_v[anchor])))
            pooled[name]["pred"].append(scores[name])
            pooled[name]["truth"].append(truth)
            pooled[name]["anchor"].extend([f"{seed}:{anchor_id}" for anchor_id in anchor_ids])
        split_rows.append(dict(seed=seed, split=split))

    # Cross-seed bootstrap: resample anchors pooled over training seeds. This
    # includes training-seed variability but not new data draws.
    pooled_block = {name: dict(pred=np.concatenate(pooled[name]["pred"]),
                               truth=np.concatenate(pooled[name]["truth"])) for name in methods}
    pooled_ci = {name: discrimination.bootstrap_accuracy(pooled_block[name]["pred"],
                                                         pooled_block[name]["truth"], min_gap,
                                                         n_boot, bootstrap_seed)
                 for name in methods}
    resolved = [name for name in methods if pooled_ci[name]["ci_low"] is not None]
    for position, first in enumerate(resolved):
        for second in resolved[position + 1:]:
            difference = discrimination.bootstrap_paired_difference(
                pooled_block[first]["pred"], pooled_block[second]["pred"],
                pooled_block[first]["truth"], min_gap, n_boot, bootstrap_seed)
            comparison_rows.append(dict(method_a=first, method_b=second, **difference))

    summary = []
    for name in methods:
        subset = [row for row in metric_rows if row["method"] == name]
        accuracies = [row["accuracy"] for row in subset if row["accuracy"] is not None]
        ratios = [row["margin_ratio_median"] for row in subset if row["margin_ratio_median"] is not None]
        margins = [row["abs_margin_error_mean"] for row in subset if row["abs_margin_error_mean"] is not None]
        summary.append(dict(
            method=name, family=next(spec["family"] for spec in specs if spec["name"] == name),
            ranking_accuracy_mean=float(np.mean(accuracies)) if accuracies else None,
            ranking_accuracy_std_across_seeds=float(np.std(accuracies)) if accuracies else None,
            pooled_accuracy=pooled_ci[name]["accuracy"],
            pooled_ci_low=pooled_ci[name]["ci_low"], pooled_ci_high=pooled_ci[name]["ci_high"],
            pair_coverage_mean=float(np.mean([row["pair_coverage"] for row in subset])),
            advantage_rmse_mean=float(np.mean([row["advantage_rmse"] for row in subset])),
            margin_ratio_median_mean=float(np.mean(ratios)) if ratios else None,
            abs_margin_error_mean=float(np.mean(margins)) if margins else None,
            rank_flip_rate_mean=float(np.mean([row["rank_flip_rate"] for row in subset])),
            mean_score_std_mean=float(np.mean([row["mean_score_std"] for row in subset])),
            per_seed=[dict(seed=row["seed"], accuracy=row["accuracy"],
                           pair_coverage=row["pair_coverage"],
                           rank_flip_rate=row["rank_flip_rate"]) for row in subset],
            seeds=len(subset), anchors_per_seed=anchors, repeats=repeats))

    report = dict(
        config=dict(seeds=list(seeds), episodes_per_seed=episodes, updates=updates,
                    max_test_anchors_per_seed=anchors, score_branches_per_candidate=repeats,
                    gamma=gamma, min_gap=min_gap, bootstrap_samples=n_boot,
                    bootstrap_seed=bootstrap_seed, candidates="all actions of the toy MDP",
                    branch_seed_scheme="SeedSequence(seed,anchor,repeat,9876)"),
        estimators=specs, summary=summary, comparisons=comparison_rows,
        runs=metric_rows,
        limitations=[
            "Toy finite-horizon MDP only; no LIBERO or RLinf claim.",
            "Exact DP labels are evaluation-only and never supervise V or Q.",
            "Q-V is a comparison method, not a ground-truth judge.",
            "TD and value-difference scores read future observations; Q-V does not.",
            "One anchor per held-out episode, so bootstrap resamples anchors, not families.",
            "Candidates stay inside the reference policy's action support.",
        ])
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(
        dict(config=report["config"], estimators=specs, summary=summary,
             comparisons=comparison_rows, limitations=report["limitations"]), indent=2),
        encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(split_rows, indent=2), encoding="utf-8")
    for name, rows in (("candidate_scores.csv", candidate_rows),
                       ("metrics.csv", metric_rows),
                       ("anchor_metrics.csv", anchor_rows),
                       ("pair_metrics.csv", pair_rows),
                       ("stability.csv", stability_rows),
                       ("gap_bins.csv", gap_rows),
                       ("comparison.csv", comparison_rows)):
        _write_csv(output / name, rows)
    plot(output, summary)
    return report


def plot(output, summary):
    """Optional figure; absence of matplotlib must not fail the experiment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        labels = [row["method"] for row in summary]
        values = [row["pooled_accuracy"] or 0.0 for row in summary]
        errors = [[0 if row["pooled_accuracy"] is None or row["pooled_ci_low"] is None
                   else row["pooled_accuracy"] - row["pooled_ci_low"] for row in summary],
                  [0 if row["pooled_accuracy"] is None or row["pooled_ci_high"] is None
                   else row["pooled_ci_high"] - row["pooled_accuracy"] for row in summary]]
        axes[0].bar(range(len(labels)), values, yerr=errors, capsize=4)
        axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axes[0].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axes[0].set(ylabel="Paired ranking accuracy (95% cluster bootstrap)",
                    title="Local action discrimination")
        axes[1].bar(range(len(labels)), [row["rank_flip_rate_mean"] for row in summary],
                    color="tab:orange")
        axes[1].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axes[1].set(ylabel="Rank flip rate across repeats", title="Ordering stability")
        figure.tight_layout()
        figure.savefig(output / "discrimination.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
