"""Option-3 study: continuous score vs binary classification vs pairwise ranking.

All three representations share features, capacity, update budget and the
training split. They are compared on three questions:

  accuracy        how well the score matches the true advantage (continuous)
                  and how well it classifies "better than reference" (binary)
  ranking         pairwise accuracy and top-1 selection at the same state
  generalisation  seen cells vs held-out (state, remaining) cells, and a
                  training-data-fraction curve

The common cross-representation metrics are top-1 selection and regret. A
continuous regressor's RMSE and a classifier's AUROC are not comparable with
each other, so they are reported as per-representation diagnostics only.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .schema import split_families
from .synthetic import generate, exact_values, POLICY_ADVANCE
from .labels import (build_transitions, binary_labels, all_cells, anchor_truth,
                     transition_pairs, split_cells)
from .objectives import fit, Predictor, KINDS
from .metrics import (auroc, binary_metrics, selection_metrics, bootstrap,
                      bootstrap_difference)
from . import discrimination


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _subset(transition, indices):
    return {key: value[indices] for key, value in transition.items()}


def _score_anchors(predictor, anchors, n_actions):
    obs = np.array([[anchor["state"]] for anchor in anchors], dtype=np.float64)
    remaining = np.array([anchor["remaining"] for anchor in anchors], dtype=np.float64)
    scores = np.zeros((len(anchors), n_actions))
    for action in range(n_actions):
        action_matrix = np.full((len(anchors), 1), float(action))
        scores[:, action] = predictor.score(obs, remaining, action_matrix)
    return scores


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


def experiment(output, seeds=(0, 1, 2), episodes=450, updates=1000, gamma=0.9,
               horizon=24, states=5, holdout=0.25, fractions=(1.0,), min_gap=0.005,
               n_boot=2000, bootstrap_seed=0, lr=3e-4, hidden=(256, 256),
               representations=KINDS, split_seed=0, batch_size=128,
               policy_advance=POLICY_ADVANCE, regime="state_dependent"):
    import torch
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique seeds required")
    if episodes < 30 or updates < 1 or n_boot < 1 or batch_size < 1:
        raise ValueError("positive budgets, at least 30 episodes required")
    if not 0 <= gamma <= 1 or horizon < 1 or states < 1:
        raise ValueError("invalid gamma, horizon or state count")
    if not 0 <= holdout < 1:
        raise ValueError("holdout must lie in [0, 1)")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("fractions must lie in (0, 1]")
    for representation in representations:
        if representation not in KINDS:
            raise ValueError(f"unknown representation: {representation}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    rows, comparisons, anchor_rows, split_rows = [], [], [], []
    for seed in seeds:
        data = generate(episodes, seed, horizon, states, policy_advance, regime)
        split = split_families(data, seed)
        train_episodes = [e for e in data if e.episode_id in split["train"]]
        oracle_v, oracle_q = exact_values(gamma, horizon, states, policy_advance, regime)
        n_actions = oracle_q.shape[2]
        transition = build_transitions(train_episodes, oracle_q, oracle_v, horizon)
        cells = all_cells(states, horizon)
        seen_cells, unseen_cells = split_cells(cells, holdout, split_seed + seed)
        seen_keys = {f"{int(a['state'])}:{int(a['remaining'])}" for a in seen_cells}
        keys = np.array([f"{int(s)}:{int(r)}" for s, r in
                         zip(transition["state_index"], transition["remaining"])], dtype=object)
        train_indices = np.flatnonzero(np.array([key in seen_keys for key in keys]))
        if len(train_indices) < 10:
            raise ValueError("not enough training transitions after the cell split")
        split_rows.append(dict(seed=seed, seen_cells=len(seen_cells),
                               unseen_cells=len(unseen_cells),
                               train_transitions=int(len(train_indices)),
                               holdout=holdout))

        splits = (("seen", seen_cells), ("unseen", unseen_cells))
        truth = {}
        for name, anchors in splits:
            if not anchors:
                raise ValueError(f"split {name} has no anchors; lower the holdout fraction")
            q, v, advantage = anchor_truth(anchors, oracle_q, oracle_v, horizon, n_actions)
            labels = binary_labels(advantage.ravel(), min_gap)
            truth[name] = dict(q=q, v=v, advantage=advantage, labels=labels,
                               ids=[f"{int(a['state'])}:{int(a['remaining'])}" for a in anchors])

        for fraction in fractions:
            rng = np.random.default_rng([seed, int(round(fraction * 1000))])
            take = min(len(train_indices), max(10, int(round(fraction * len(train_indices)))))
            chosen = rng.choice(train_indices, size=take, replace=False)
            subset = _subset(transition, chosen)
            pairs = transition_pairs(subset, min_gap)
            if "pairwise" in representations and not pairs:
                raise ValueError("no same-cell action pairs in the training subset")

            top1_by_split = {split_name: {} for split_name, _ in splits}
            regret_by_split = {split_name: {} for split_name, _ in splits}
            for representation in representations:
                checkpoint = fit(representation, subset, pairs=pairs, horizon=horizon,
                                 updates=updates, batch_size=batch_size, seed=seed,
                                 lr=lr, hidden=hidden, min_gap=min_gap)
                path = output / f"{representation}_f{int(round(fraction*100))}_s{seed}.pt"
                torch.save(checkpoint, path)
                predictor = Predictor(path)

                for name, anchors in splits:
                    reference = truth[name]
                    count = len(anchors)
                    scores = _score_anchors(predictor, anchors, n_actions)
                    records = discrimination.pairwise_records(
                        scores, reference["q"], min_gap, reference["ids"])
                    pairwise = discrimination.aggregate_metrics(records)
                    selection = selection_metrics(scores, reference["q"], min_gap)

                    classification = dict(auroc=None, average_precision=None, brier=None,
                                          ece=None, threshold_accuracy=None,
                                          positive_rate=None)
                    if representation == "binary":
                        classification = binary_metrics(reference["labels"],
                                                        _sigmoid(scores.ravel()))
                    error = dict(advantage_rmse=None, normalized_rmse=None,
                                 calibration_slope=None)
                    if representation == "continuous":
                        error = discrimination.advantage_error_metrics(
                            scores, reference["advantage"])

                    def pairwise_at(index, _scores=scores, _q=reference["q"],
                                    _ids=reference["ids"]):
                        subset_ids = [_ids[i] for i in index]
                        subset_records = discrimination.pairwise_records(
                            _scores[index], _q[index], min_gap, subset_ids)
                        return discrimination.aggregate_metrics(subset_records)["accuracy"]
                    def top1_at(index, _scores=scores, _q=reference["q"]):
                        return selection_metrics(_scores[index], _q[index], min_gap)["top1"]
                    def regret_at(index, _scores=scores, _q=reference["q"]):
                        return selection_metrics(_scores[index], _q[index], min_gap)["regret"]
                    def auroc_at(index, _scores=scores, _labels=reference["labels"]):
                        flat = _scores[index].ravel()
                        flat_labels = _labels.reshape(_scores.shape)[index].ravel()
                        return auroc(flat_labels, _sigmoid(flat))
                    def rmse_at(index, _scores=scores, _advantage=reference["advantage"]):
                        return float(np.sqrt(np.mean((_scores[index] - _advantage[index]) ** 2)))

                    pair_ci = bootstrap(pairwise_at, count, n_boot, bootstrap_seed)
                    top1_ci = bootstrap(top1_at, count, n_boot, bootstrap_seed)
                    regret_ci = bootstrap(regret_at, count, n_boot, bootstrap_seed)
                    auroc_ci = (bootstrap(auroc_at, count, n_boot, bootstrap_seed)
                                if representation == "binary"
                                else dict(ci_low=None, ci_high=None))
                    rmse_ci = (bootstrap(rmse_at, count, n_boot, bootstrap_seed)
                               if representation == "continuous"
                               else dict(ci_low=None, ci_high=None))
                    prevalence = float(np.mean(reference["labels"][reference["labels"] >= 0] == 1))

                    rows.append(dict(seed=seed, representation=representation,
                                     fraction=fraction, split=name, anchors=count,
                                     train_transitions=take, pairs=len(pairs),
                                     pairwise_accuracy=pairwise["accuracy"],
                                     pairwise_ci_low=pair_ci["ci_low"],
                                     pairwise_ci_high=pair_ci["ci_high"],
                                     top1=selection["top1"],
                                     top1_ci_low=top1_ci["ci_low"],
                                     top1_ci_high=top1_ci["ci_high"],
                                     regret=selection["regret"],
                                     regret_ci_low=regret_ci["ci_low"],
                                     regret_ci_high=regret_ci["ci_high"],
                                     auroc=classification["auroc"],
                                     auroc_ci_low=auroc_ci["ci_low"],
                                     auroc_ci_high=auroc_ci["ci_high"],
                                     average_precision=classification["average_precision"],
                                     brier=classification["brier"], ece=classification["ece"],
                                     threshold_accuracy=classification["threshold_accuracy"],
                                     positive_rate=prevalence,
                                     advantage_rmse=error["advantage_rmse"],
                                     rmse_ci_low=rmse_ci["ci_low"], rmse_ci_high=rmse_ci["ci_high"],
                                     normalized_rmse=error["normalized_rmse"],
                                     calibration_slope=error["calibration_slope"]))
                    anchor_rows.extend(dict(seed=seed, representation=representation,
                                            fraction=fraction, split=name,
                                            anchor_id=reference["ids"][index],
                                            action=action, score=float(scores[index, action]),
                                            true_advantage=float(reference["advantage"][index, action]))
                                       for index in range(count) for action in range(n_actions))
                    chosen_action = scores.argmax(axis=1)
                    regret = reference["q"].max(axis=1) - reference["q"][np.arange(count), chosen_action]
                    top1_by_split[name][representation] = (regret <= min_gap).astype(float)
                    regret_by_split[name][representation] = regret

            for name, _ in splits:
                baseline_top1 = top1_by_split[name].get("continuous")
                baseline_regret = regret_by_split[name].get("continuous")
                if baseline_top1 is None:
                    continue
                for candidate in representations:
                    if candidate == "continuous":
                        continue
                    top1_diff = bootstrap_difference(top1_by_split[name][candidate],
                                                     baseline_top1, n_boot, bootstrap_seed)
                    regret_diff = bootstrap_difference(regret_by_split[name][candidate],
                                                       baseline_regret, n_boot, bootstrap_seed)
                    comparisons.append(dict(seed=seed, fraction=fraction, split=name,
                                            representation=candidate, baseline="continuous",
                                            delta_top1=top1_diff["delta"],
                                            delta_top1_ci_low=top1_diff["ci_low"],
                                            delta_top1_ci_high=top1_diff["ci_high"],
                                            delta_regret=regret_diff["delta"],
                                            delta_regret_ci_low=regret_diff["ci_low"],
                                            delta_regret_ci_high=regret_diff["ci_high"]))

    summary = []
    keys = sorted({(row["representation"], row["fraction"], row["split"]) for row in rows})
    for representation, fraction, name in keys:
        group = [row for row in rows
                 if (row["representation"], row["fraction"], row["split"]) ==
                 (representation, fraction, name)]
        def mean(key):
            values = [row[key] for row in group if row[key] is not None]
            return float(np.mean(values)) if values else None
        def std(key):
            values = [row[key] for row in group if row[key] is not None]
            return float(np.std(values)) if values else None
        summary.append(dict(representation=representation, fraction=fraction, split=name,
                            pairwise_accuracy_mean=mean("pairwise_accuracy"),
                            pairwise_accuracy_std=std("pairwise_accuracy"),
                            top1_mean=mean("top1"), top1_std=std("top1"),
                            regret_mean=mean("regret"), regret_std=std("regret"),
                            auroc_mean=mean("auroc"), average_precision_mean=mean("average_precision"),
                            brier_mean=mean("brier"), ece_mean=mean("ece"),
                            threshold_accuracy_mean=mean("threshold_accuracy"),
                            advantage_rmse_mean=mean("advantage_rmse"),
                            normalized_rmse_mean=mean("normalized_rmse"),
                            calibration_slope_mean=mean("calibration_slope"),
                            seeds=len(group)))

    report = dict(config=dict(seeds=list(seeds), episodes_per_seed=episodes, updates=updates,
                              gamma=gamma, horizon=horizon, states=states,
                              holdout=holdout, fractions=list(fractions),
                              min_gap=min_gap, bootstrap_samples=n_boot,
                              bootstrap_seed=bootstrap_seed, learning_rate=lr,
                              regime=regime,
                              hidden=[int(units) for units in hidden],
                              representations=list(representations),
                              split_seed=split_seed,
                              splits="train episodes by family; (state, remaining) cells held out",
                              candidates="all actions of the toy MDP"),
                  summary=summary, comparisons=comparisons,
                  limitations=[
                      "Toy finite-horizon MDP only; no LIBERO or RLinf claim.",
                      "Binary and pairwise labels come from the exact oracle; on real data they require independent rollouts.",
                      "Top-1 and regret are the only metrics comparable across representations; AUROC and RMSE are diagnostics.",
                      "Held-out cells test unseen regions of a five-state MDP; this is a weak notion of generalisation.",
                      "The training budget, features and capacity are held fixed; a difference is not a tuning result.",
                  ])
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(split_rows, indent=2), encoding="utf-8")
    _write_csv(output / "cells.csv", rows)
    _write_csv(output / "comparisons.csv", comparisons)
    _write_csv(output / "anchor_scores.csv", anchor_rows)
    plot(output, summary)
    return report


def plot(output, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        splits = sorted({row["split"] for row in summary})
        fractions = sorted({row["fraction"] for row in summary})
        representations = sorted({row["representation"] for row in summary})
        figure, axes = plt.subplots(1, max(len(fractions), 1),
                                    figsize=(4.5 * max(len(fractions), 1), 3.6),
                                    squeeze=False)
        for position, fraction in enumerate(fractions):
            axis = axes[0][position]
            width = 0.8 / max(len(splits), 1)
            for offset, name in enumerate(splits):
                values = []
                for representation in representations:
                    match = [row for row in summary if row["representation"] == representation
                             and row["fraction"] == fraction and row["split"] == name]
                    values.append(match[0]["top1_mean"] if match else 0.0)
                positions = np.arange(len(representations)) + offset * width
                axis.bar(positions, values, width=width, label=name)
            axis.set_xticks(np.arange(len(representations)) + width, representations, rotation=20)
            axis.set(ylim=(0, 1), title=f"train fraction {fraction}", ylabel="top-1")
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / "top1_by_representation.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
