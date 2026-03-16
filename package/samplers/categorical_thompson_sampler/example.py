"""
Example for the CategoricalThompsonSampler.

Demonstrates both ``mode="independent"`` and ``mode="gibbs"``.

**Single categorical variable** (independent mode):
    Three categories produce noisy rewards from Gaussians with different
    means/widths.  Category *c* should usually win when maximizing.

**Two correlated categorical variables** (Gibbs mode):
    The reward depends on whether ``color`` and ``size`` are a "good"
    combination (e.g. ``("red", "large")``), demonstrating that Gibbs
    conditioning captures interactions that independent sampling misses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import optuna
import optunahub


# ── Shared helpers ────────────────────────────────────────────────────

package_name = "package/samplers/categorical_thompson_sampler"
test_local = True
registry_root = "/Users/sammcdermott/local_git/optunahub-registry"


def _load_sampler(**kwargs: Any) -> Any:
    if test_local:
        mod = optunahub.load_local_module(package=package_name, registry_root=registry_root)
    else:
        mod = optunahub.load_module(package=package_name)
    return mod.CategoricalThompsonSampler(**kwargs)


def _summarize(
    study: optuna.study.Study,
    cat_param: str,
    categories: list[str],
) -> str:
    cat_results: dict[str, list[float]] = defaultdict(list)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            cat_results[trial.params[cat_param]].append(trial.value)
    return ", ".join(
        f"{c}: {len(cat_results[c])} trials (best {max(cat_results[c]):.2f})"
        for c in categories
        if c in cat_results
    )


# ── Example 1: single categorical variable (independent mode) ────────


def objective_single(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    cat = trial.suggest_categorical("cat", ["a", "b", "c"])
    rng = np.random.RandomState(trial.number)
    if cat == "a":
        return x + rng.normal(loc=3.0, scale=0.5)
    elif cat == "b":
        return x + rng.normal(loc=0.0, scale=2.0)
    else:
        return x + rng.normal(loc=2.0, scale=2.0)


def run_independent_example() -> None:
    print("=" * 60)
    print("Example 1: single categorical — independent mode")
    print("=" * 60)

    n_runs, n_trials = 20, 30
    win_counts: dict[str, int] = defaultdict(int)

    for run in range(n_runs):
        sampler = _load_sampler(burn_in=4, seed=run, mode="independent")
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective_single, n_trials=n_trials, show_progress_bar=False)

        best_cat = study.best_trial.params["cat"]
        win_counts[best_cat] += 1
        summary = _summarize(study, "cat", ["a", "b", "c"])
        print(f"Run {run + 1:2d}: winner = {best_cat}  |  {summary}")

    print(f"\n--- Summary over {n_runs} runs ---")
    for c in ["a", "b", "c"]:
        print(f"  '{c}' won {win_counts[c]}/{n_runs} times")


# ── Example 2: two correlated categoricals (Gibbs mode) ──────────────

GOOD_COMBOS = {("red", "large"), ("blue", "small"), ("green", "medium")}


def objective_correlated(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    color = trial.suggest_categorical("color", ["red", "blue", "green"])
    size = trial.suggest_categorical("size", ["small", "medium", "large"])
    rng = np.random.RandomState(trial.number)
    bonus = 4.0 if (color, size) in GOOD_COMBOS else 0.0
    return x + rng.normal(loc=bonus, scale=1.0)


def run_gibbs_example() -> None:
    print("\n" + "=" * 60)
    print("Example 2: two correlated categoricals — Gibbs vs independent")
    print("=" * 60)

    n_runs, n_trials = 20, 60

    for mode in ("independent", "gibbs"):
        good_combo_wins = 0
        best_values: list[float] = []

        for run in range(n_runs):
            sampler = _load_sampler(burn_in=3, seed=run, mode=mode)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective_correlated, n_trials=n_trials, show_progress_bar=False)

            bt = study.best_trial
            best_values.append(bt.value)
            combo = (bt.params["color"], bt.params["size"])
            if combo in GOOD_COMBOS:
                good_combo_wins += 1

        avg_best = np.mean(best_values)
        print(
            f"  {mode:12s}: good-combo best in {good_combo_wins}/{n_runs} runs, "
            f"avg best value = {avg_best:.2f}"
        )


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_independent_example()
    run_gibbs_example()
