from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import optuna


logger = logging.getLogger(__name__)

# Key prefix for storing per-category samples in study user_attrs.
_SAMPLES_ATTR_PREFIX = "categorical_thompson_samples:"


class CategoricalThompsonSampler(optuna.samplers.BaseSampler):
    """Sampler that uses Thompson sampling for categorical variables.

    This subclasses ``optuna.samplers.BaseSampler`` to add Thompson sampling
    for categorical variables while delegating numerical parameters to a
    ``base_sampler``.

    Categorical parameters are excluded from the relative search space so that
    they are handled via ``sample_independent``, where the Thompson sampling
    logic lives.  A burn-in phase cycles through each category a fixed number
    of times before switching to Thompson sampling.

    This version works only for a single categorical variable but could be
    extended if desired.
    """

    def __init__(
        self,
        burn_in: int = 10,
        base_sampler: Optional[optuna.samplers.BaseSampler] = None,
        categorical_variable_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            burn_in: Number of times each category is sampled during burn-in.
            base_sampler: Sampler used for numerical parameters.  Defaults to
                ``TPESampler`` if *None*.
            categorical_variable_name: Name of the categorical parameter.  If
                *None* it is discovered automatically during sampling.
            seed: Random seed for the Thompson sampling RNG.
        """
        self.base_sampler = base_sampler or optuna.samplers.TPESampler()
        self.burn_in = burn_in
        self._burning_in = True
        self.categorical_variable_name = categorical_variable_name
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Search-space: exclude categorical distributions so they fall
    # through to sample_independent.
    # ------------------------------------------------------------------

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        search_space = self.base_sampler.infer_relative_search_space(study, trial)
        # Remove categorical distributions – they will be handled in
        # sample_independent via Thompson sampling.
        return {
            name: dist
            for name, dist in search_space.items()
            if not isinstance(dist, optuna.distributions.CategoricalDistribution)
        }

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        # search_space already has categoricals removed by
        # infer_relative_search_space, so we can delegate directly.
        return self.base_sampler.sample_relative(study, trial, search_space)

    # ------------------------------------------------------------------
    # Independent sampling: Thompson sampling for categoricals.
    # ------------------------------------------------------------------

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        if isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            return self._sample_categorical(study, param_name, param_distribution)
        return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

    # ------------------------------------------------------------------
    # Core Thompson sampling logic.
    # ------------------------------------------------------------------

    def _sample_categorical(
        self,
        study: optuna.study.Study,
        param_name: str,
        param_distribution: optuna.distributions.CategoricalDistribution,
    ) -> Any:
        """Thompson sampling for a single categorical parameter.

        During burn-in, cycles through every category ``self.burn_in`` times.
        Afterwards, draws one stored observation at random for each category
        and picks the best (respecting ``study.direction``).
        """
        categories = param_distribution.choices
        n_categories = len(categories)

        # Retrieve stored per-category samples for *this* parameter.
        samples = self._get_samples(study, param_name)

        # --- Burn-in phase ---
        total_burn_in_trials = self.burn_in * n_categories
        n_observed = sum(len(v) for v in samples.values())

        if n_observed < total_burn_in_trials:
            return categories[n_observed % n_categories]

        # --- Thompson sampling phase ---
        if self._burning_in:
            logger.info(
                "%d burns for each of the %d categories of '%s' completed. "
                "Moving to Thompson sampling.",
                self.burn_in,
                n_categories,
                param_name,
            )
            self._burning_in = False

        # Only consider categories that belong to *this* parameter.
        # For each category, draw one stored value uniformly at random.
        maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

        best_category = None
        best_value = None
        for cat in categories:
            cat_samples = samples.get(cat, [])
            if not cat_samples:
                continue
            drawn = self._rng.choice(cat_samples)
            if (
                best_value is None
                or (maximize and drawn > best_value)
                or (not maximize and drawn < best_value)
            ):
                best_value = drawn
                best_category = cat

        if best_category is None:
            # Fallback: no samples yet (shouldn't happen after burn-in).
            return self._rng.choice(categories)

        return best_category

    # ------------------------------------------------------------------
    # after_trial: record the objective value for the chosen category.
    # ------------------------------------------------------------------

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if state != optuna.trial.TrialState.COMPLETE or values is None:
            return

        for param_name, distribution in trial.distributions.items():
            if not isinstance(distribution, optuna.distributions.CategoricalDistribution):
                continue

            category = trial.params[param_name]
            value = values[0]

            # Store the observation using study user_attrs.
            attr_key = f"{_SAMPLES_ATTR_PREFIX}{param_name}:{category}"
            existing = study.user_attrs.get(attr_key, [])
            existing.append(value)
            study.set_user_attr(attr_key, existing)

        self.base_sampler.after_trial(study, trial, state, values)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_samples(study: optuna.study.Study, param_name: str) -> Dict[str, list]:
        """Retrieve per-category samples from study user_attrs."""
        prefix = f"{_SAMPLES_ATTR_PREFIX}{param_name}:"
        samples: Dict[str, list] = {}
        for key, value in study.user_attrs.items():
            if key.startswith(prefix):
                category = key[len(prefix) :]
                samples[category] = value
        return samples
