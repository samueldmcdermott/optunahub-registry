"""
Example for the CategoricalThompsonSampler.

Three categories produce noisy objective values drawn from Gaussians:
  a: narrow Gaussian at +3  (mu=3, sigma=0.5) — high mean but little upside
  b: wide Gaussian at 0     (mu=0, sigma=2)   — centered at zero, rarely large
  c: wide Gaussian at +2    (mu=2, sigma=2)   — decent mean with large upside

Since we maximize, c should usually win (shifted right with wide tails gives
the best draws), a sometimes wins (reliable but capped), and b rarely if ever
wins.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    cat = trial.suggest_categorical("cat", ["a", "b", "c"])
    rng = np.random.RandomState(trial.number)
    if cat == "a":
        return x + rng.normal(loc=3.0, scale=0.5)
    elif cat == "b":
        return x + rng.normal(loc=0.0, scale=2.0)
    else:
        return x + rng.normal(loc=2.0, scale=2.0)


package_name = "package/samplers/categorical_thompson_sampler"
test_local = True

n_runs = 20
n_trials = 30
win_counts: dict[str, int] = defaultdict(int)

for run in range(n_runs):
    if test_local:
        sampler = optunahub.load_local_module(
            package=package_name,
            registry_root="/Users/sammcdermott/local_git/optunahub-registry",
        ).CategoricalThompsonSampler(burn_in=4, seed=run)
    else:
        sampler = optunahub.load_module(
            package=package_name,
        ).CategoricalThompsonSampler(burn_in=4, seed=run)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_cat = study.best_trial.params["cat"]
    win_counts[best_cat] += 1

    # Per-category breakdown for this run.
    cat_results: dict[str, list[float]] = defaultdict(list)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            cat_results[trial.params["cat"]].append(trial.value)

    cats_summary = ", ".join(
        f"{c}: {len(cat_results[c])} trials (best {max(cat_results[c]):.2f})"
        for c in ["a", "b", "c"]
        if c in cat_results
    )
    print(f"Run {run + 1:2d}: winner = {best_cat}  |  {cats_summary}")

print(f"\n--- Summary over {n_runs} runs ---")
for c in ["a", "b", "c"]:
    print(f"  '{c}' won {win_counts[c]}/{n_runs} times")
