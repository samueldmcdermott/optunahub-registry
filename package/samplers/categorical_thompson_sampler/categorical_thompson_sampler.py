from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set

import numpy as np
import optuna


logger = logging.getLogger(__name__)


class CategoricalThompsonSampler(optuna.samplers.BaseSampler):
    """Sampler that uses Thompson sampling for categorical variables.

    This subclasses ``optuna.samplers.BaseSampler`` to add Thompson sampling
    for categorical variables while delegating numerical parameters to a
    ``base_sampler``.

    Two modes are supported:

    ``"independent"``
        Each categorical parameter is sampled independently.  Fast and simple,
        but ignores interactions between categorical parameters.

    ``"gibbs"``
        All categorical parameters are sampled together via a Gibbs-style
        conditional sweep.  On each trial a single sweep cycles through every
        categorical parameter, sampling it conditioned on the current values of
        all other categorical parameters.  This captures interactions between
        parameters while scaling linearly in the total number of categories.
        When too few conditioned observations are available, the sampler falls
        back to the marginal (unconditional) reward history automatically.

    In both modes a burn-in phase cycles through each category a fixed number
    of times before switching to Thompson sampling.
    """

    def __init__(
        self,
        burn_in: int = 10,
        base_sampler: Optional[optuna.samplers.BaseSampler] = None,
        seed: Optional[int] = None,
        mode: str = "independent",
        gibbs_min_samples: int = 2,
    ) -> None:
        """
        Args:
            burn_in: Number of times each category is sampled during burn-in.
            base_sampler: Sampler used for numerical parameters.  Defaults to
                ``TPESampler`` if *None*.
            seed: Random seed for the Thompson sampling RNG.
            mode: ``"independent"`` or ``"gibbs"``.
            gibbs_min_samples: In Gibbs mode, the minimum number of
                conditioned observations required before using conditional
                samples.  When fewer are available the marginal (unconditional)
                history is used instead.
        """
        if mode not in ("independent", "gibbs"):
            raise ValueError(f"mode must be 'independent' or 'gibbs', got {mode!r}")

        self.base_sampler = base_sampler or optuna.samplers.TPESampler()
        self.burn_in = burn_in
        self.mode = mode
        self._gibbs_min_samples = gibbs_min_samples
        self._rng = np.random.RandomState(seed)
        self._burn_in_logged: Set[str] = set()

    # ------------------------------------------------------------------
    # Search-space & relative sampling
    # ------------------------------------------------------------------

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        search_space = self.base_sampler.infer_relative_search_space(study, trial)

        if self.mode == "gibbs":
            # Keep categoricals in the relative search space so they are
            # handled together in sample_relative.
            return search_space

        # Independent mode: remove categoricals so they fall through to
        # sample_independent.
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
        if self.mode == "gibbs":
            categorical_space = {
                n: d
                for n, d in search_space.items()
                if isinstance(d, optuna.distributions.CategoricalDistribution)
            }
            numerical_space = {
                n: d
                for n, d in search_space.items()
                if not isinstance(d, optuna.distributions.CategoricalDistribution)
            }

            numerical_values = self.base_sampler.sample_relative(study, trial, numerical_space)

            if categorical_space:
                categorical_values = self._gibbs_sweep(study, categorical_space)
                return {**numerical_values, **categorical_values}
            return numerical_values

        # Independent mode: search_space already has categoricals removed.
        return self.base_sampler.sample_relative(study, trial, search_space)

    # ------------------------------------------------------------------
    # Independent sampling
    # ------------------------------------------------------------------

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        if isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            samples = self._get_marginal_samples(study, param_name)
            return self._thompson_sample_param(
                study, param_name, param_distribution.choices, samples
            )
        return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

    # ------------------------------------------------------------------
    # Gibbs sweep
    # ------------------------------------------------------------------

    def _gibbs_sweep(
        self,
        study: optuna.study.Study,
        categorical_space: Dict[str, optuna.distributions.CategoricalDistribution],
    ) -> Dict[str, Any]:
        """Sample all categorical parameters via one Gibbs-conditional sweep.

        During burn-in every parameter independently cycles through its
        categories.  Once all parameters have completed burn-in, a single
        Gibbs sweep is performed: for each parameter in turn, we condition
        on the current values of all other parameters and Thompson-sample
        from the filtered reward history.
        """
        # --- Check per-parameter burn-in ---
        all_past_burn_in = True
        values: Dict[str, Any] = {}
        for param_name, dist in categorical_space.items():
            categories = dist.choices
            samples = self._get_marginal_samples(study, param_name)
            n_observed = sum(len(v) for v in samples.values())

            if n_observed < self.burn_in * len(categories):
                values[param_name] = categories[n_observed % len(categories)]
                all_past_burn_in = False
            else:
                values[param_name] = None  # placeholder; filled below
                self._log_burn_in_complete(param_name, len(categories))

        if not all_past_burn_in:
            # Some params are still burning in — use independent Thompson
            # sampling for the rest.
            for param_name, dist in categorical_space.items():
                if values[param_name] is None:
                    samples = self._get_marginal_samples(study, param_name)
                    values[param_name] = self._pick_best_category(study, dist.choices, samples)
            return values

        # --- Full Gibbs sweep ---
        current = self._initialize_from_best_trial(study, categorical_space)
        for param_name, dist in categorical_space.items():
            other_values = {k: v for k, v in current.items() if k != param_name}
            conditioned = self._get_conditioned_samples(study, param_name, other_values)
            total_conditioned = sum(len(v) for v in conditioned.values())

            if total_conditioned < self._gibbs_min_samples:
                # Not enough conditioned data — fall back to marginal.
                conditioned = self._get_marginal_samples(study, param_name)

            current[param_name] = self._pick_best_category(study, dist.choices, conditioned)
        return current

    # ------------------------------------------------------------------
    # Core Thompson sampling logic
    # ------------------------------------------------------------------

    def _thompson_sample_param(
        self,
        study: optuna.study.Study,
        param_name: str,
        categories: tuple,  # type: ignore[type-arg]
        samples: Dict[Any, List[float]],
    ) -> Any:
        """Burn-in + Thompson sampling for a single categorical parameter."""
        n_categories = len(categories)
        n_observed = sum(len(v) for v in samples.values())

        if n_observed < self.burn_in * n_categories:
            return categories[n_observed % n_categories]

        self._log_burn_in_complete(param_name, n_categories)
        return self._pick_best_category(study, categories, samples)

    def _pick_best_category(
        self,
        study: optuna.study.Study,
        categories: tuple,  # type: ignore[type-arg]
        samples: Dict[Any, List[float]],
    ) -> Any:
        """Draw one stored value per category and return the best."""
        maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
        best_category = None
        best_value: Optional[float] = None
        for cat in categories:
            cat_samples = samples.get(cat, [])
            if not cat_samples:
                continue
            drawn = float(self._rng.choice(cat_samples))
            if (
                best_value is None
                or (maximize and drawn > best_value)
                or (not maximize and drawn < best_value)
            ):
                best_value = drawn
                best_category = cat

        if best_category is None:
            return categories[int(self._rng.randint(len(categories)))]
        return best_category

    # ------------------------------------------------------------------
    # after_trial
    # ------------------------------------------------------------------

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self.base_sampler.after_trial(study, trial, state, values)

    # ------------------------------------------------------------------
    # Helpers — trial-history scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _get_marginal_samples(
        study: optuna.study.Study, param_name: str
    ) -> Dict[Any, List[float]]:
        """Reward samples grouped by category (unconditional)."""
        samples: Dict[Any, List[float]] = {}
        for trial in study.trials:
            if (
                trial.state != optuna.trial.TrialState.COMPLETE
                or param_name not in trial.params
                or trial.value is None
            ):
                continue
            cat = trial.params[param_name]
            samples.setdefault(cat, []).append(trial.value)
        return samples

    @staticmethod
    def _get_conditioned_samples(
        study: optuna.study.Study,
        param_name: str,
        conditions: Dict[str, Any],
    ) -> Dict[Any, List[float]]:
        """Reward samples for *param_name*, keeping only trials where every
        parameter in *conditions* matches its specified value."""
        samples: Dict[Any, List[float]] = {}
        for trial in study.trials:
            if (
                trial.state != optuna.trial.TrialState.COMPLETE
                or param_name not in trial.params
                or trial.value is None
            ):
                continue
            if any(trial.params.get(k) != v for k, v in conditions.items()):
                continue
            cat = trial.params[param_name]
            samples.setdefault(cat, []).append(trial.value)
        return samples

    @staticmethod
    def _initialize_from_best_trial(
        study: optuna.study.Study,
        categorical_space: Dict[str, optuna.distributions.CategoricalDistribution],
    ) -> Dict[str, Any]:
        """Seed the Gibbs sweep with the best trial's categorical values."""
        completed = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if not completed:
            # Should not happen after burn-in, but be safe.
            return {name: dist.choices[0] for name, dist in categorical_space.items()}

        maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
        best = (
            max(completed, key=lambda t: t.value)
            if maximize
            else min(completed, key=lambda t: t.value)
        )  # type: ignore[arg-type,return-value]

        values: Dict[str, Any] = {}
        for name, dist in categorical_space.items():
            if name in best.params:
                values[name] = best.params[name]
            else:
                values[name] = dist.choices[0]
        return values

    def _log_burn_in_complete(self, param_name: str, n_categories: int) -> None:
        if param_name not in self._burn_in_logged:
            logger.info(
                "%d burns for each of the %d categories of '%s' completed. "
                "Moving to Thompson sampling.",
                self.burn_in,
                n_categories,
                param_name,
            )
            self._burn_in_logged.add(param_name)
