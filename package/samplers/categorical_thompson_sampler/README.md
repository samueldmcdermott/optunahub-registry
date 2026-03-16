---
author: Samuel D. McDermott
title: Categorical Thompson Sampler
description: Sampler based on Thompson sampling for categorical variables.
tags: [sampler, Thompson sampling, categorical variables]
optuna_versions: [4.2.1]
license: MIT License
---

## Class or Function Names

- CategoricalThompsonSampler

## Example

### Independent mode (default) — single categorical variable

```python
import numpy as np
import optuna
import optunahub

def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    cat = trial.suggest_categorical("cat", ["a", "b", "c"])
    rng = np.random.RandomState(trial.number)
    if cat == "a":
        return x + rng.normal(loc=3.0, scale=0.5)
    elif cat == "b":
        return x + rng.normal(loc=0.0, scale=2.0)
    else:
        return x + rng.normal(loc=2.0, scale=2.0)

sampler = optunahub.load_module(
    package="package/samplers/categorical_thompson_sampler",
).CategoricalThompsonSampler(burn_in=4, mode="independent")

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=30)
print(study.best_trial.params)
```

### Gibbs mode — multiple correlated categorical variables

When the objective depends on *interactions* between categorical parameters,
Gibbs-conditional Thompson sampling captures those correlations by sampling
each parameter conditioned on the current values of the others.

```python
import numpy as np
import optuna
import optunahub

GOOD_COMBOS = {("red", "large"), ("blue", "small"), ("green", "medium")}

def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    color = trial.suggest_categorical("color", ["red", "blue", "green"])
    size = trial.suggest_categorical("size", ["small", "medium", "large"])
    rng = np.random.RandomState(trial.number)
    bonus = 4.0 if (color, size) in GOOD_COMBOS else 0.0
    return x + rng.normal(loc=bonus, scale=1.0)

sampler = optunahub.load_module(
    package="package/samplers/categorical_thompson_sampler",
).CategoricalThompsonSampler(burn_in=3, mode="gibbs")

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=60)
print(study.best_trial.params)
```

## Others

This package provides a sampler based on the principles of Thompson sampling.
For a pedagogical introduction, see
[A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038).

**Modes:**

- `"independent"` (default) — each categorical parameter is sampled
  independently.  Fast and simple; works well when categorical parameters do
  not interact.
- `"gibbs"` — all categorical parameters are sampled together via a
  Gibbs-style conditional sweep.  On each trial the sampler cycles through
  every categorical parameter, sampling it conditioned on the current values
  of the others.  Falls back to marginal (unconditional) sampling
  automatically when conditioned data is sparse.
