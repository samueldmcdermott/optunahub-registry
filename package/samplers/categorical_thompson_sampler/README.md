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

```python
from collections import defaultdict

import numpy as np
import optuna
import optunahub

def gaussians(x: float, label: str):
    if label == 'a':
        return np.random.normal(loc=1, scale=8)
    elif label == 'b':
        return np.random.normal(loc=5, scale=2)
    elif label == 'c':
        return np.random.normal(loc=0, scale=3)
    else:
        return np.random.normal(loc=2, scale=2)

def objective(trial):
    xv = trial.suggest_float('x', -1, 1)
    label = trial.suggest_categorical('label', ['a', 'b', 'c', 'd'])
    return gaussians(xv, label)


package_name = "package/samplers/categorical_thompson_sampler"
sampler = optunahub.load_module(
    package=package_name,
).CategoricalThompsonSampler()

study_T = optuna.create_study(direction='maximize', sampler=sampler)
study_T.optimize(objective, n_trials=111)

study_base = optuna.create_study(direction='maximize')
study_base.optimize(objective, n_trials=111)

# Compare per-category results.
thompson_cats = defaultdict(list)
base_cats = defaultdict(list)
for t in study_T.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        thompson_cats[t.params['label']].append(t.value)
for t in study_base.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        base_cats[t.params['label']].append(t.value)

for k in sorted(thompson_cats):
    print(f"label {k}:")
    print(f"\tThompson sampler: max = {max(thompson_cats[k]):.3f} from {len(thompson_cats[k])} samples")
    print(f"\tBase sampler: max = {max(base_cats[k]):.3f} from {len(base_cats[k])} samples")
```

The base sampler follows a "winner takes all" approach, whereas Thompson sampling does a better job of balancing exploration and exploitation.

## Others

This package provides a sampler based on the principles of Thompson sampling. For a pedagogical introduction, see [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038).
