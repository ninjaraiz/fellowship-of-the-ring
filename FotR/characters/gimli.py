"""
gimli.py
=========

This module provides a unified interface for several unsupervised
learning algorithms used in CFD post-processing. It replaces the
previous monolithic ``SAM.Weapons.GMM()`` static method with a
small, extensible object-oriented framework built around two core
ideas:

1.  **Configuration is immutable.** :class:`ClusterConfig` is a frozen
    dataclass. Once a :class:`GIMLI` instance is built, its
    configuration cannot be mutated in place — any change goes through
    :meth:`GIMLI.reconfigure`, which builds a brand-new
    ``ClusterConfig`` via ``dataclasses.replace()`` and resets every
    piece of generated state. This removes an entire class of bugs
    caused by shared mutable configuration objects, and makes it
    trivial to repeat an analysis with a different configuration
    without worrying about leftover state from a previous run.

2.  **All generated state lives in GIMLI, never in the configuration.**
    ``dataset``, ``results`` and ``summary_table`` are attributes of
    the :class:`GIMLI` instance. ``ClusterConfig`` only ever describes
    *how* an analysis should be run, never *what* has been computed.

Currently supported algorithms:

    - Gaussian Mixture Models (``"gmm"``) — fully implemented.

Planned (the :class:`BaseClusterModel` contract is ready for them,
only the concrete subclasses are missing):

    - K-Means
    - DBSCAN
    - HDBSCAN
    - Agglomerative Clustering
    - Spectral Clustering
    - Birch

Adding a new algorithm only requires subclassing :class:`BaseClusterModel`
and decorating it with :func:`register_model`; no change to :class:`GIMLI`
itself is needed (factory / registry pattern).

Typical workflow
-----------------

::

    cluster = GIMLI(
        df,
        variables=["Cp", "Cf"],
        groups=["AoA", "Mach"],
        algorithm="gmm",
        cluster_range=range(1, 7),
        covariance_types=("diag", "full"),
        output_dir="./gimli_study",
    )

    cluster.prepare()
    cluster.select_model()      # BIC/AIC sweep per group
    cluster.fit()                # uses the recommended n_components per group
    cluster.predict()            # labels + membership probabilities
    cluster.summary()            # one row per group with metrics
    cluster.save()               # tables, figures and (optionally) models

    cluster.plot.bic(group=(2.0, 0.7))
    cluster.plot.scatter(group=(2.0, 0.7))
    cluster.plot.heatmap(value="recommended_n")

Repeating the analysis with a different configuration::

    cluster.reconfigure(scaler="robust", covariance_types=("full",))
    cluster.prepare().select_model().fit().predict().summary()
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import os

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import numpy as np
import pandas as pd

from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# ============================================================================
# Module-level registries
# ----------------------------------------------------------------------------
# These dictionaries live at the top of the module (rather than next to the
# classes that populate them) because the registration decorator needs the
# target dict to already exist by the time the decorated classes are parsed.
# Both registries are looked up only at *runtime* (inside method bodies), by
# which point the whole module has finished importing, so classes defined
# later in this same file (e.g. GMMModel) can safely be referenced from
# classes defined earlier (e.g. GIMLI, ClusterConfig).
# ============================================================================

MODEL_REGISTRY: dict[str, type["BaseClusterModel"]] = {}

_SCALER_REGISTRY: dict[str, type] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}

# Direction in which each model-selection metric improves. Used by
# GIMLI.select_model() to decide whether to take the row with the minimum
# or the maximum value of `config.selection` from a model's selection table.
_METRIC_DIRECTION: dict[str, str] = {
    "bic": "min",
    "aic": "min",
    "log_likelihood": "max",
    "silhouette": "max",
    "calinski_harabasz": "max",
    "davies_bouldin": "min",
}


def register_model(name: str):
    """
    Class decorator that registers a :class:`BaseClusterModel` subclass
    under ``name`` in :data:`MODEL_REGISTRY`, so that
    ``ClusterConfig(algorithm=name)`` and ``GIMLI(..., algorithm=name)``
    can find and instantiate it.

    This implements a factory / registry pattern: adding a new clustering
    algorithm to the framework never requires touching :class:`GIMLI` or
    :class:`ClusterConfig` — only a new subclass decorated with
    ``@register_model("my_algo")`` needs to be written.

    Parameters
    ----------
    name : str
        The string identifier users will pass as ``algorithm=name``.

    Returns
    -------
    Callable
        A decorator that registers the class and returns it unchanged.

    Examples
    --------
    ::

        @register_model("kmeans")
        class KMeansModel(BaseClusterModel):
            name = "kmeans"
            supports_probabilities = False
            supports_model_selection = True
            supports_covariance = False

            def select_model(self, X, config):
                ...

            def fit(self, X, config, n_clusters, covariance_type=None):
                ...

            def predict(self, model, X):
                return model.predict(X)

        # Now usable directly:
        cluster = GIMLI(df, variables=["Cp"], algorithm="kmeans")
    """

    def _decorator(cls: type["BaseClusterModel"]) -> type["BaseClusterModel"]:
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


# ============================================================================
# GIMLI — main user-facing class
# ============================================================================

class GIMLI:
    """
    Generic clustering framework with a stable, six-method public API.

    GIMLI orchestrates a full clustering study (optionally split into
    independent groups, e.g. one CFD case per AoA/Mach combination) on
    top of any algorithm registered in :data:`MODEL_REGISTRY`. It owns
    every piece of state generated by the analysis (``dataset``,
    ``results``, ``summary_table``); the analysis *settings* live in
    the immutable :attr:`config` attribute and are never touched by
    GIMLI itself.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain every column listed in ``variables``
        and, if given, every column listed in ``groups``. A defensive
        copy is stored internally; the caller's DataFrame is never
        mutated.
    variables : list[str]
        Feature columns used for clustering (e.g. ``["Cp", "Cf"]``).
    groups : list[str] or None
        Column(s) used to split ``df`` into independent groups that
        are clustered separately (e.g. ``["AoA", "Mach"]``). If
        ``None`` or empty, the whole DataFrame is treated as a single
        group.
    config : ClusterConfig or None
        A pre-built immutable configuration. Mutually exclusive with
        ``**kwargs``.
    **kwargs
        Forwarded to ``ClusterConfig(...)`` when ``config`` is not
        given. See :class:`ClusterConfig` for every available field.

    Attributes
    ----------
    config : ClusterConfig
        Immutable analysis configuration. Replace it via
        :meth:`reconfigure`, never by assigning to its fields directly
        (which would raise ``dataclasses.FrozenInstanceError``).
    dataset : ClusterDataset or None
        Populated by :meth:`prepare`.
    results : dict[tuple, ClusterResult]
        One entry per group, keyed by the group's groupby key
        (an empty tuple ``()`` when ``groups`` is empty).
    summary_table : pd.DataFrame or None
        Populated by :meth:`summary`.
    plot : ClusterPlotter
        Namespace with plotting helpers (``plot.scatter()``,
        ``plot.bic()``, ``plot.aic()``, ``plot.heatmap()``,
        ``plot.probabilities()``).

    Examples
    --------
    ::

        cluster = GIMLI(
            df, variables=["Cp", "Cf"], groups=["AoA", "Mach"],
            algorithm="gmm", cluster_range=range(1, 6),
        )
        cluster.prepare().select_model().fit().predict()
        print(cluster.summary())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        variables,
        groups=None,
        config: Optional["ClusterConfig"] = None,
        **kwargs,
    ):
        if config is not None and kwargs:
            raise ValueError(
                "Pass either a pre-built `config` or configuration "
                "**kwargs, not both."
            )

        self.df = df.copy()
        self.variables = list(variables)
        self.groups = [] if groups is None else list(groups)

        self.config: "ClusterConfig" = config if config is not None else ClusterConfig(**kwargs)
        self._algorithm: "BaseClusterModel" = MODEL_REGISTRY[self.config.algorithm]()

        self.dataset: Optional["ClusterDataset"] = None
        self.results: dict[tuple, "ClusterResult"] = {}
        self.summary_table: Optional[pd.DataFrame] = None

        self.plot = ClusterPlotter(self)

        self._prepared = False
        self._fitted = False

    # ------------------------------------------------------------------
    # Configuration management
    # ------------------------------------------------------------------

    def reconfigure(self, **kwargs) -> "GIMLI":
        """
        Replace the current configuration with a new one and reset all
        generated state.

        Because :class:`ClusterConfig` is frozen, this method never
        mutates the existing configuration object in place. Instead it
        builds a brand-new ``ClusterConfig`` from the current one plus
        the given overrides (via ``dataclasses.replace``) and reassigns
        :attr:`config` to point to it. ``dataset``, ``results`` and
        ``summary_table`` are cleared so that the next call to
        :meth:`prepare` starts from a clean state — this is what makes
        it safe to repeat an analysis with a different configuration
        without leftover state from the previous run.

        Parameters
        ----------
        **kwargs
            Any field of :class:`ClusterConfig` to override (e.g.
            ``scaler="robust"``, ``covariance_types=("full",)``).

        Returns
        -------
        GIMLI
            ``self``, to allow method chaining.

        Examples
        --------
        ::

            cluster.reconfigure(scaler="robust", random_state=1)
            cluster.prepare().select_model().fit()

            # Switching algorithm entirely (once more are registered):
            cluster.reconfigure(algorithm="kmeans", cluster_range=range(2, 8))
        """
        self.config = replace(self.config, **kwargs)
        self._algorithm = MODEL_REGISTRY[self.config.algorithm]()
        self._reset_state()

        if self.config.verbose:
            print(f"[GIMLI] reconfigured: {kwargs}")

        return self

    def _reset_state(self) -> None:
        """Clear every piece of state generated by a previous run."""
        self.dataset = None
        self.results = {}
        self.summary_table = None
        self._prepared = False
        self._fitted = False

    # ------------------------------------------------------------------
    # Stable public API
    # ------------------------------------------------------------------

    def prepare(self) -> "GIMLI":
        """
        Build the working dataset: extract the feature matrix, split it
        into groups, and scale each group independently according to
        ``config.scaler``.

        Scaling is performed per group (not globally) so that, e.g.,
        each AoA/Mach case is normalised on its own statistics — this
        mirrors how the previous ``SAM.Weapons.GMM()`` method behaved
        and is usually the right choice for CFD cases where different
        flow conditions can have very different magnitudes.

        Populates :attr:`dataset` with a :class:`ClusterDataset`
        instance and resets ``results``/``summary_table`` so this can
        be safely called again after :meth:`reconfigure`.

        Returns
        -------
        GIMLI
            ``self``, to allow method chaining.

        Raises
        ------
        KeyError
            If any column in ``variables`` or ``groups`` is missing
            from the input DataFrame.

        Examples
        --------
        ::

            cluster = GIMLI(df, variables=["Cp", "Cf"], groups=["AoA"])
            cluster.prepare()
            print(cluster.dataset.X_scaled.shape)
        """
        feature_cols = list(self.variables)
        required_cols = feature_cols + self.groups
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise KeyError(
                f"Columns {missing} not found in the input DataFrame."
            )

        X_full = self.df[feature_cols].to_numpy(dtype=float)

        if self.groups:
            raw_indices = self.df.groupby(self.groups).indices
            group_indices = {
                (key if isinstance(key, tuple) else (key,)): np.asarray(idx)
                for key, idx in raw_indices.items()
            }
        else:
            group_indices = {(): np.arange(len(self.df))}

        scaler_cls = _SCALER_REGISTRY.get(self.config.scaler)

        X_scaled = np.full_like(X_full, np.nan, dtype=float)
        scalers: dict[tuple, Any] = {}

        for key, idx in group_indices.items():
            X_group = X_full[idx]
            if scaler_cls is None:  # config.scaler == "none"
                X_scaled[idx] = X_group
                scalers[key] = None
            else:
                scaler = scaler_cls()
                X_scaled[idx] = scaler.fit_transform(X_group)
                scalers[key] = scaler

        self.dataset = ClusterDataset(
            dataframe=self.df,
            X=X_full,
            X_scaled=X_scaled,
            feature_names=feature_cols,
            group_columns=list(self.groups),
            group_indices=group_indices,
            scalers=scalers,
        )

        self.results = {}
        self.summary_table = None
        self._fitted = False
        self._prepared = True

        if self.config.verbose:
            print(
                f"[GIMLI] prepared {len(group_indices)} group(s), "
                f"{X_full.shape[0]} sample(s), {X_full.shape[1]} feature(s), "
                f"scaler='{self.config.scaler}'."
            )

        return self

    def select_model(self) -> "GIMLI":
        """
        Run a model-selection sweep for every group, independently.

        For each group, delegates to the configured algorithm's
        ``select_model(X, config)`` (see :class:`BaseClusterModel`),
        which returns a DataFrame with one row per combination of
        ``config.cluster_range`` × ``config.covariance_types`` (the
        latter ignored by algorithms that do not support it) and one
        column per metric the algorithm can compute (e.g. ``bic``,
        ``aic``, ``log_likelihood`` for GMM).

        The row that optimises ``config.selection`` (minimised for
        ``"bic"``/``"aic"``/``"davies_bouldin"``, maximised for
        ``"silhouette"``/``"calinski_harabasz"``/``"log_likelihood"``)
        is stored as the group's recommendation
        (``ClusterResult.recommended_n`` /
        ``ClusterResult.recommended_covariance``), which :meth:`fit`
        uses automatically unless explicitly overridden.

        Groups with fewer samples than ``max(config.cluster_range)``
        are skipped with a warning (when ``config.verbose=True``) and
        left without a recommendation; :meth:`fit` will require an
        explicit ``n_clusters`` for them.

        When ``config.n_jobs != 1``, groups are processed in parallel
        with ``joblib`` using the ``loky`` backend.

        Returns
        -------
        GIMLI
            ``self``, to allow method chaining.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare`.
        NotImplementedError
            If the configured algorithm does not support model
            selection (``supports_model_selection = False``).

        Examples
        --------
        ::

            cluster.prepare().select_model()
            print(cluster.results[(2.0, 0.7)].selection_table)
            print(cluster.results[(2.0, 0.7)].recommended_n)
        """
        self._require_prepared()

        if not self._algorithm.supports_model_selection:
            raise NotImplementedError(
                f"Algorithm '{self.config.algorithm}' does not support "
                "model selection."
            )

        min_required = max(self.config.cluster_range)
        job_specs: list[tuple[tuple, np.ndarray]] = []

        for key, idx in self.dataset.group_indices.items():
            X_group = self.dataset.X_scaled[idx]
            self.results[key] = self.results.get(key) or ClusterResult(group=key)

            if X_group.shape[0] < min_required:
                if self.config.verbose:
                    print(
                        f"[GIMLI] group {key}: only {X_group.shape[0]} sample(s), "
                        f"fewer than the largest n_components to test "
                        f"({min_required}). Skipping selection."
                    )
                continue

            job_specs.append((key, X_group))

        if self.config.n_jobs != 1 and len(job_specs) > 1:
            from joblib import Parallel, delayed

            outputs = Parallel(n_jobs=self.config.n_jobs)(
                delayed(_select_model_worker)(key, Xg, self._algorithm, self.config)
                for key, Xg in job_specs
            )
        else:
            outputs = [
                _select_model_worker(key, Xg, self._algorithm, self.config)
                for key, Xg in job_specs
            ]

        direction = _METRIC_DIRECTION[self.config.selection]

        for key, table in outputs:
            if self.config.selection not in table.columns:
                raise ValueError(
                    f"Selection criterion '{self.config.selection}' is not "
                    f"produced by algorithm '{self.config.algorithm}'. "
                    f"Available columns: {list(table.columns)}."
                )

            result = self.results[key]
            result.selection_table = table.reset_index(drop=True)

            valid = table.dropna(subset=[self.config.selection])
            if not valid.empty:
                best_idx = (
                    valid[self.config.selection].idxmin()
                    if direction == "min"
                    else valid[self.config.selection].idxmax()
                )
                best_row = valid.loc[best_idx]
                result.recommended_n = int(best_row["n_components"])
                result.recommended_covariance = best_row.get("covariance_type")

            self.results[key] = result

            if self.config.verbose:
                extra = (
                    f", covariance_type='{result.recommended_covariance}'"
                    if result.recommended_covariance
                    else ""
                )
                print(
                    f"[GIMLI] group {key}: recommended n_components="
                    f"{result.recommended_n}{extra} (by {self.config.selection})."
                )

        return self

    def fit(
        self,
        n_clusters: Optional[int] = None,
        covariance_type: Optional[str] = None,
    ) -> "GIMLI":
        """
        Fit the final clustering model for every group.

        The number of clusters and the covariance type are resolved,
        per group, in this order of priority:

        1. The explicit ``n_clusters`` / ``covariance_type`` arguments
           (applied identically to every group).
        2. The recommendation produced by a previous call to
           :meth:`select_model` for that specific group.
        3. For ``covariance_type`` only: ``config.covariance_types[0]``
           if exactly one covariance type was configured.

        If neither (1) nor (2) can resolve ``n_clusters`` for a group,
        a ``RuntimeError`` is raised naming that group. This is a
        deliberate stable contract: GIMLI never silently guesses a
        number of clusters.

        Groups with fewer samples than the resolved ``n_clusters`` are
        skipped with a warning (when ``config.verbose=True``); their
        :class:`ClusterResult` is left with ``model=None``, which later
        steps (e.g. :meth:`predict`) skip gracefully.

        When ``config.n_jobs != 1``, groups are fitted in parallel with
        ``joblib`` using the ``loky`` backend. Only the per-group
        feature matrix, the (stateless) algorithm wrapper and the
        configuration are sent to worker processes — never the whole
        ``GIMLI`` instance or the original DataFrame — to keep
        inter-process communication cheap.

        Parameters
        ----------
        n_clusters : int or None
            Override the per-group recommendation for every group.
        covariance_type : str or None
            Override the per-group recommendation for every group.
            Ignored by algorithms with ``supports_covariance = False``.

        Returns
        -------
        GIMLI
            ``self``, to allow method chaining.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare`, or if a group has no
            resolvable ``n_clusters``/``covariance_type``.

        Examples
        --------
        ::

            # Use the BIC-recommended n_components per group:
            cluster.prepare().select_model().fit()

            # Force the same n_clusters everywhere, skipping selection:
            cluster.prepare().fit(n_clusters=3, covariance_type="full")
        """
        self._require_prepared()

        job_specs: list[tuple[tuple, np.ndarray, int, Optional[str]]] = []

        for key, idx in self.dataset.group_indices.items():
            result = self.results.get(key) or ClusterResult(group=key)

            n_use = n_clusters if n_clusters is not None else result.recommended_n
            if n_use is None:
                raise RuntimeError(
                    f"No n_clusters available for group {key!r}. Either pass "
                    "n_clusters explicitly to fit(), or run select_model() "
                    "first."
                )

            cov_use = covariance_type or result.recommended_covariance
            if self._algorithm.supports_covariance and cov_use is None:
                if len(self.config.covariance_types) == 1:
                    cov_use = self.config.covariance_types[0]
                else:
                    raise RuntimeError(
                        f"No covariance_type available for group {key!r}. "
                        "Either pass covariance_type explicitly to fit(), "
                        "or run select_model() first."
                    )

            X_group = self.dataset.X_scaled[idx]
            if X_group.shape[0] < n_use:
                if self.config.verbose:
                    print(
                        f"[GIMLI] group {key}: only {X_group.shape[0]} "
                        f"sample(s), fewer than n_clusters={n_use}. Skipping fit."
                    )
                self.results[key] = result
                continue

            result.n_clusters_used = n_use
            result.covariance_type_used = cov_use
            self.results[key] = result
            job_specs.append((key, X_group, n_use, cov_use))

        if self.config.n_jobs != 1 and len(job_specs) > 1:
            from joblib import Parallel, delayed

            outputs = Parallel(n_jobs=self.config.n_jobs)(
                delayed(_fit_group_worker)(
                    key, Xg, self._algorithm, self.config, n_use, cov_use
                )
                for key, Xg, n_use, cov_use in job_specs
            )
        else:
            outputs = [
                _fit_group_worker(key, Xg, self._algorithm, self.config, n_use, cov_use)
                for key, Xg, n_use, cov_use in job_specs
            ]

        for key, model, metrics in outputs:
            self.results[key].model = model
            self.results[key].metrics = metrics

        self._fitted = True

        if self.config.verbose:
            n_ok = sum(1 for r in self.results.values() if r.model is not None)
            print(
                f"[GIMLI] fitted {n_ok}/{len(self.results)} group(s) "
                f"using algorithm '{self.config.algorithm}'."
            )

        return self

    def predict(self) -> "GIMLI":
        """
        Compute hard labels (and, when supported, soft membership
        probabilities) for every fitted group, and attach them to the
        working dataframe.

        Two columns are added to ``self.dataset.dataframe``:

        - ``"cluster"`` : int — the assigned cluster label. Labels are
          local to each group: cluster ``0`` in one group has no
          relationship to cluster ``0`` in another group, since each
          group is fitted independently.
        - ``"cluster_proba_max"`` : float — the maximum posterior
          membership probability for the assigned cluster (i.e. how
          confident the model is about that point's assignment). Only
          filled in for algorithms with ``supports_probabilities=True``
          (e.g. GMM); left as ``NaN`` otherwise.

        The full probability matrix (one column per component) is
        *not* merged into the dataframe, because different groups may
        have a different number of clusters and therefore a different
        number of probability columns — it remains available per group
        as ``self.results[group].probabilities``.

        Returns
        -------
        GIMLI
            ``self``, to allow method chaining.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        Examples
        --------
        ::

            cluster.prepare().select_model().fit().predict()
            df_out = cluster.dataset.dataframe
            print(df_out[["cluster", "cluster_proba_max"]].head())

            # Per-group full probability matrix:
            proba = cluster.results[(2.0, 0.7)].probabilities
        """
        self._require_fitted()

        df_out = self.dataset.dataframe.copy()
        df_out["cluster"] = -1
        df_out["cluster_proba_max"] = np.nan

        cluster_col = df_out.columns.get_loc("cluster")
        proba_col = df_out.columns.get_loc("cluster_proba_max")

        for key, idx in self.dataset.group_indices.items():
            result = self.results.get(key)
            if result is None or result.model is None:
                continue

            X_group = self.dataset.X_scaled[idx]
            labels = self._algorithm.predict(result.model, X_group)
            result.labels = labels
            df_out.iloc[idx, cluster_col] = labels

            if self._algorithm.supports_probabilities:
                proba = self._algorithm.predict_proba(result.model, X_group)
                result.probabilities = proba
                df_out.iloc[idx, proba_col] = proba.max(axis=1)

            self.results[key] = result

        self.dataset.dataframe = df_out

        if self.config.verbose:
            n_labelled = int((df_out["cluster"] != -1).sum())
            print(f"[GIMLI] predicted labels for {n_labelled}/{len(df_out)} row(s).")

        return self

    def summary(self) -> pd.DataFrame:
        """
        Build (and return) a one-row-per-group summary table.

        Each row contains the group's identifying columns (exploded
        from the groupby key), the number of samples, the number of
        clusters actually used, the covariance type used (if
        applicable), the model-selection recommendation (if
        :meth:`select_model` was run), and every metric recorded in
        :class:`ClusterMetrics` (``bic``, ``aic``, ``silhouette``,
        ``calinski_harabasz``, ``davies_bouldin``, ``log_likelihood``,
        ``entropy``, ``converged``, ``iterations``).

        The result is stored in :attr:`summary_table` in addition to
        being returned, and printed when ``config.verbose=True``.

        Returns
        -------
        pd.DataFrame
            One row per group.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        Examples
        --------
        ::

            cluster.prepare().select_model().fit()
            table = cluster.summary()
            print(table.sort_values("bic").head())
        """
        self._require_fitted()

        records = []
        for key, result in self.results.items():
            row: dict[str, Any] = {}
            if self.groups:
                for col, val in zip(self.groups, key):
                    row[col] = val

            row["n_samples"] = int(len(self.dataset.group_indices[key]))
            row["n_clusters_used"] = result.n_clusters_used
            row["covariance_type_used"] = result.covariance_type_used
            row["recommended_n"] = result.recommended_n
            row["recommended_covariance"] = result.recommended_covariance

            m = result.metrics
            row.update(
                {
                    "bic": m.bic,
                    "aic": m.aic,
                    "log_likelihood": m.log_likelihood,
                    "silhouette": m.silhouette,
                    "calinski_harabasz": m.calinski_harabasz,
                    "davies_bouldin": m.davies_bouldin,
                    "entropy": m.entropy,
                    "converged": m.converged,
                    "iterations": m.iterations,
                }
            )
            records.append(row)

        self.summary_table = pd.DataFrame.from_records(records)

        if self.config.verbose:
            print(self.summary_table.to_string(index=False))

        return self.summary_table

    def save(self) -> "GIMLI":
        """
        Persist tables, figures and (optionally) fitted models to
        ``config.output_dir``, according to ``config.save_tables``,
        ``config.save_figures`` and ``config.save_models``.

        Layout written under ``config.output_dir``::

            tables/clustered_data.csv     (only if save_tables)
            tables/summary.csv            (only if save_tables)
            tables/selection_tables.csv   (only if save_tables and any
                                            group ran select_model())
            figures/scatter_<group>.png   (only if save_figures)
            figures/bic_<group>.png       (only if save_figures and the
                                            group ran select_model())
            figures/heatmap_recommended_n.png
                                           (only if save_figures and
                                            exactly two `groups` columns)
            models/model_<group>.joblib   (only if save_models)

        A single figure failing to render (e.g. a group with too few
        points for a 2-D scatter) only emits a warning when
        ``config.verbose=True`` and never aborts the rest of
        :meth:`save`.

        Returns
        -------
        GIMLI
            ``self``, to allow method chaining.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If any ``save_*`` flag is ``True`` but ``config.output_dir``
            is ``None``.

        Examples
        --------
        ::

            cluster.reconfigure(output_dir="./gimli_study")
            cluster.prepare().select_model().fit().predict()
            cluster.save()
        """
        self._require_fitted()

        needs_dir = (
            self.config.save_tables
            or self.config.save_figures
            or self.config.save_models
        )
        if needs_dir and self.config.output_dir is None:
            raise ValueError(
                "config.output_dir must be set to save tables, figures or "
                "models. Use gimli.reconfigure(output_dir='...')."
            )

        out_dir = self.config.output_dir

        if self.config.save_tables:
            self._save_tables(out_dir)

        if self.config.save_figures:
            self._save_figures(out_dir)

        if self.config.save_models:
            self._save_models(out_dir)

        return self

    # ------------------------------------------------------------------
    # save() helpers
    # ------------------------------------------------------------------

    def _save_tables(self, out_dir: str) -> None:
        """Write clustered data, summary and selection tables to CSV."""
        tables_dir = os.path.join(out_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        self.dataset.dataframe.to_csv(
            os.path.join(tables_dir, "clustered_data.csv"), sep=";", index=True
        )

        if self.summary_table is None:
            self.summary()
        self.summary_table.to_csv(
            os.path.join(tables_dir, "summary.csv"), sep=";", index=False
        )

        selection_frames = [
            r.selection_table for r in self.results.values()
            if r.selection_table is not None
        ]
        if selection_frames:
            pd.concat(selection_frames, ignore_index=True).to_csv(
                os.path.join(tables_dir, "selection_tables.csv"),
                sep=";",
                index=False,
            )

        if self.config.verbose:
            print(f"[GIMLI] tables saved in {tables_dir}")

    def _save_figures(self, out_dir: str) -> None:
        """Render and save the per-group and global diagnostic figures."""
        figures_dir = os.path.join(out_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        for key in self.results:
            label = _group_label(key)

            try:
                self.plot.scatter(
                    group=key,
                    save_path=os.path.join(figures_dir, f"scatter_{label}.png"),
                )
                plt.close("all")
            except Exception as exc:
                if self.config.verbose:
                    print(f"[GIMLI] scatter plot skipped for group {key}: {exc}")

            if self.results[key].selection_table is not None:
                try:
                    self.plot.bic(
                        group=key,
                        save_path=os.path.join(figures_dir, f"bic_{label}.png"),
                    )
                    plt.close("all")
                except Exception as exc:
                    if self.config.verbose:
                        print(f"[GIMLI] bic plot skipped for group {key}: {exc}")

        if len(self.groups) == 2:
            try:
                self.plot.heatmap(
                    value="recommended_n",
                    save_path=os.path.join(figures_dir, "heatmap_recommended_n.png"),
                )
                plt.close("all")
            except Exception as exc:
                if self.config.verbose:
                    print(f"[GIMLI] heatmap skipped: {exc}")

        if self.config.verbose:
            print(f"[GIMLI] figures saved in {figures_dir}")

    def _save_models(self, out_dir: str) -> None:
        """Pickle every fitted model to disk via joblib."""
        import joblib

        models_dir = os.path.join(out_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        for key, result in self.results.items():
            if result.model is None:
                continue
            label = _group_label(key)
            joblib.dump(result.model, os.path.join(models_dir, f"model_{label}.joblib"))

        if self.config.verbose:
            print(f"[GIMLI] models saved in {models_dir}")

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _require_prepared(self) -> None:
        """Raise if prepare() has not been called yet."""
        if not self._prepared or self.dataset is None:
            raise RuntimeError("Call prepare() before this operation.")

    def _require_fitted(self) -> None:
        """Raise if fit() has not been called yet."""
        self._require_prepared()
        if not self._fitted:
            raise RuntimeError("Call fit() before this operation.")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else ("prepared" if self._prepared else "unprepared")
        return (
            f"<GIMLI algorithm='{self.config.algorithm}' status={status} "
            f"groups={self.groups or 'none'} n_results={len(self.results)}>"
        )


# ============================================================================
# Module-level worker functions (used for both sequential and joblib-parallel
# execution). Defined at module scope — and taking only small, explicit
# arguments rather than closing over `self` — so that parallel workers never
# need to pickle the whole GIMLI instance (in particular, its DataFrame).
# ============================================================================

def _select_model_worker(
    key: tuple,
    X_group: np.ndarray,
    algorithm: "BaseClusterModel",
    config: "ClusterConfig",
) -> tuple[tuple, pd.DataFrame]:
    """Run one group's model-selection sweep; safe to call from joblib."""
    table = algorithm.select_model(X_group, config).copy()
    table.insert(0, "group", [key] * len(table))
    return key, table


def _fit_group_worker(
    key: tuple,
    X_group: np.ndarray,
    algorithm: "BaseClusterModel",
    config: "ClusterConfig",
    n_clusters: int,
    covariance_type: Optional[str],
) -> tuple[tuple, Any, "ClusterMetrics"]:
    """Fit one group's final model; safe to call from joblib."""
    model, metrics = algorithm.fit(X_group, config, n_clusters, covariance_type)
    return key, model, metrics


def _group_label(key: tuple) -> str:
    """
    Build a filesystem-safe label for a group key, used to name output
    files. Numeric components are formatted with limited precision so
    that filenames stay short and readable; an empty key (the
    ungrouped case) becomes ``"global"``.
    """
    if not key:
        return "global"
    parts = [f"{v:.3g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v) for v in key]
    return "_".join(parts)


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """
    Immutable configuration of a clustering analysis.

    Every field below is read-only after construction: attempting
    ``config.algorithm = "kmeans"`` raises
    ``dataclasses.FrozenInstanceError``. To change settings, build a
    new configuration — either directly (``ClusterConfig(...)``) or,
    when working through a :class:`GIMLI` instance, via
    :meth:`GIMLI.reconfigure`, which also resets the generated state
    that depended on the old configuration.

    Parameters
    ----------
    algorithm : str
        Key into :data:`MODEL_REGISTRY` selecting the clustering
        algorithm. Default ``"gmm"``.
    scaler : str
        One of ``"standard"``, ``"robust"``, ``"minmax"`` or
        ``"none"``. Applied independently to each group in
        :meth:`GIMLI.prepare`. ``"robust"`` (median / IQR) is usually
        preferable to ``"standard"`` when features contain outliers,
        e.g. pressure coefficients near a shock. Default ``"standard"``.
    random_state : int
        Seed forwarded to every stochastic estimator for
        reproducibility. Default ``0``.
    verbose : bool
        Print progress information from every public method. Default
        ``True``.
    n_jobs : int
        Number of groups to process in parallel (via ``joblib``) in
        :meth:`GIMLI.select_model` and :meth:`GIMLI.fit`. ``1`` runs
        sequentially (default); ``-1`` uses every available core.
    cluster_range : range
        Candidate numbers of clusters swept by :meth:`GIMLI.select_model`.
        Default ``range(2, 10)``.
    selection : str
        Metric used to pick the recommended number of clusters from the
        selection table. One of ``"bic"``, ``"aic"``,
        ``"log_likelihood"``, ``"silhouette"``, ``"calinski_harabasz"``
        or ``"davies_bouldin"``. Default ``"bic"``.
    covariance_types : tuple[str, ...]
        Covariance structures swept by algorithms with
        ``supports_covariance = True`` (currently GMM). Any subset of
        ``("full", "tied", "diag", "spherical")``. Default
        ``("diag",)``.
    max_iter : int
        Maximum iterations for the underlying estimator. Default ``500``.
    tol : float
        Convergence tolerance for the underlying estimator. Default
        ``1e-3``.
    n_init : int
        Number of independent initialisations for the underlying
        estimator (the best one is kept). Default ``5``.
    save_figures : bool
        Whether :meth:`GIMLI.save` writes diagnostic figures. Default
        ``True``.
    save_tables : bool
        Whether :meth:`GIMLI.save` writes CSV tables. Default ``True``.
    save_models : bool
        Whether :meth:`GIMLI.save` pickles fitted models with
        ``joblib``. Default ``False`` (models can be large).
    output_dir : str or None
        Destination directory for everything written by
        :meth:`GIMLI.save`. Required (non-``None``) whenever any of
        the three ``save_*`` flags above is ``True``.

    Examples
    --------
    ::

        # Direct construction:
        config = ClusterConfig(
            algorithm="gmm", scaler="robust",
            cluster_range=range(1, 8), covariance_types=("diag", "full"),
            output_dir="./gimli_study",
        )
        cluster = GIMLI(df, variables=["Cp", "Cf"], config=config)

        # Equivalent, via GIMLI's **kwargs forwarding:
        cluster = GIMLI(
            df, variables=["Cp", "Cf"], scaler="robust",
            cluster_range=range(1, 8), covariance_types=("diag", "full"),
            output_dir="./gimli_study",
        )

        # Deriving a variant configuration without touching the original:
        from dataclasses import replace
        config_robust = replace(config, scaler="robust")
    """

    # -------------------------
    # General
    # -------------------------

    algorithm: str = "gmm"

    scaler: str = "standard"

    random_state: int = 0

    verbose: bool = True

    n_jobs: int = 1

    # -------------------------
    # Model selection
    # -------------------------

    cluster_range: range = field(
        default_factory=lambda: range(2, 10)
    )

    selection: str = "bic"

    covariance_types: tuple[str, ...] = (
        "diag",
    )

    # -------------------------
    # Fit
    # -------------------------

    max_iter: int = 500

    tol: float = 1e-3

    n_init: int = 5

    # -------------------------
    # Output
    # -------------------------

    save_figures: bool = True

    save_tables: bool = True

    save_models: bool = False

    output_dir: Optional[str] = None

    def __post_init__(self):
        """Validate field values eagerly, at construction time."""
        if self.algorithm not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Available: {sorted(MODEL_REGISTRY)}."
            )
        if self.scaler not in _SCALER_REGISTRY and self.scaler != "none":
            raise ValueError(
                f"Unknown scaler '{self.scaler}'. "
                f"Available: {sorted(_SCALER_REGISTRY)} or 'none'."
            )
        if self.selection not in _METRIC_DIRECTION:
            raise ValueError(
                f"Unknown selection criterion '{self.selection}'. "
                f"Available: {sorted(_METRIC_DIRECTION)}."
            )
        if self.n_jobs == 0:
            raise ValueError(
                "n_jobs cannot be 0. Use 1 for sequential execution or "
                "-1 to use every available core."
            )
        if len(self.covariance_types) == 0:
            raise ValueError("covariance_types must contain at least one value.")
        if len(list(self.cluster_range)) == 0:
            raise ValueError("cluster_range must contain at least one value.")


# ============================================================================
# Generated state (owned by GIMLI, never by ClusterConfig)
# ============================================================================

@dataclass(slots=True)
class ClusterMetrics:
    """
    Metrics describing the quality of a single fitted clustering model.

    Fields are deliberately generic so that every algorithm in the
    registry can populate the subset that applies to it and leave the
    rest as ``None`` (e.g. K-Means has no ``bic``/``aic`` in the
    classical sense, GMM has no ``inertia``).

    Attributes
    ----------
    bic, aic : float or None
        Bayesian / Akaike information criteria (model-based, lower is
        better). Populated by GMM.
    silhouette, calinski_harabasz, davies_bouldin : float or None
        Model-agnostic geometric cluster-quality metrics computed from
        the feature matrix and the hard labels (silhouette and
        calinski_harabasz: higher is better; davies_bouldin: lower is
        better). Computed by :meth:`BaseClusterModel.compute_generic_metrics`
        and shared across every algorithm.
    log_likelihood : float or None
        Total log-likelihood of the data under the fitted model
        (higher is better). Populated by likelihood-based models.
    entropy : float or None
        Average Shannon entropy of the per-sample membership
        probabilities (lower means more confident assignments).
        Populated by algorithms with soft assignments (e.g. GMM).
    inertia : float or None
        Sum of squared distances to the nearest cluster centre.
        Populated by centroid-based models (e.g. K-Means).
    converged : bool or None
        Whether the underlying optimisation reported convergence.
    iterations : int or None
        Number of iterations the underlying optimisation ran for.
    """

    bic: float | None = None

    aic: float | None = None

    silhouette: float | None = None

    calinski_harabasz: float | None = None

    davies_bouldin: float | None = None

    log_likelihood: float | None = None

    entropy: float | None = None

    inertia: float | None = None

    converged: bool | None = None

    iterations: int | None = None


@dataclass(slots=True)
class ClusterDataset:
    """
    The prepared, scaled dataset GIMLI clusters on, as built by
    :meth:`GIMLI.prepare`.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The working copy of the input data. :meth:`GIMLI.predict`
        replaces this with a version carrying the added
        ``"cluster"``/``"cluster_proba_max"`` columns.
    X : np.ndarray
        Raw (unscaled) feature matrix, shape ``(n_samples, n_features)``,
        in the same row order as ``dataframe``.
    X_scaled : np.ndarray or None
        Feature matrix after per-group scaling, same shape as ``X``.
    feature_names : list[str]
        Column names corresponding to the columns of ``X``/``X_scaled``.
    group_columns : list[str]
        Column names used to split the data into groups (empty if the
        whole dataset is treated as a single group).
    group_indices : dict[tuple, np.ndarray]
        Maps each group key (an empty tuple ``()`` for the ungrouped
        case) to the integer row positions belonging to that group.
    scalers : dict[tuple, Any]
        Maps each group key to the fitted scaler instance used for
        that group (``None`` when ``config.scaler == "none"``). Kept
        around in case the caller needs to transform new data back
        into the original scale, or apply the exact same
        transformation to additional points later.
    """

    dataframe: pd.DataFrame

    X: np.ndarray

    X_scaled: np.ndarray | None = None

    feature_names: list[str] = field(
        default_factory=list
    )

    group_columns: list[str] = field(
        default_factory=list
    )

    group_indices: dict = field(
        default_factory=dict
    )

    scalers: dict = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ClusterResult:
    """
    Everything GIMLI knows about a single group's clustering, from
    model selection through to predicted labels.

    Attributes
    ----------
    group : tuple
        The group's groupby key (``()`` for the ungrouped case).
    model : Any or None
        The fitted estimator object (e.g. a ``sklearn.mixture.GaussianMixture``)
        returned by :meth:`BaseClusterModel.fit`. ``None`` until
        :meth:`GIMLI.fit` succeeds for this group.
    labels : np.ndarray or None
        Hard cluster assignments, populated by :meth:`GIMLI.predict`.
    probabilities : np.ndarray or None
        Soft membership probabilities, shape
        ``(n_samples_in_group, n_clusters_used)``, populated by
        :meth:`GIMLI.predict` for algorithms with
        ``supports_probabilities = True``.
    metrics : ClusterMetrics
        Quality metrics of the fitted model.
    selection_table : pd.DataFrame or None
        The full model-selection sweep table for this group, populated
        by :meth:`GIMLI.select_model`.
    recommended_n : int or None
        Number of clusters recommended by :meth:`GIMLI.select_model`
        according to ``config.selection``.
    recommended_covariance : str or None
        Covariance type recommended by :meth:`GIMLI.select_model`
        (only meaningful for algorithms with ``supports_covariance = True``).
    n_clusters_used : int or None
        Number of clusters actually used by :meth:`GIMLI.fit` for this
        group (the resolved value, after applying explicit overrides
        or the selection recommendation).
    covariance_type_used : str or None
        Covariance type actually used by :meth:`GIMLI.fit` for this
        group.
    dataframe : pd.DataFrame or None
        The slice of the working dataframe belonging to this group,
        with predicted columns attached (populated by
        :meth:`GIMLI.predict`).
    """

    group: tuple

    model: Any = None

    labels: np.ndarray | None = None

    probabilities: np.ndarray | None = None

    metrics: ClusterMetrics = field(
        default_factory=ClusterMetrics
    )

    selection_table: pd.DataFrame | None = None

    recommended_n: int | None = None

    recommended_covariance: str | None = None

    n_clusters_used: int | None = None

    covariance_type_used: str | None = None

    dataframe: pd.DataFrame | None = None


# ============================================================================
# Algorithm contract
# ============================================================================

class BaseClusterModel(ABC):
    """
    Abstract interface every clustering algorithm in the registry must
    implement.

    A subclass is a thin, *stateless* wrapper around a clustering
    library (typically scikit-learn): it never stores fitted state on
    ``self`` between calls — the fitted estimator object is always
    returned to (and owned by) the caller (:class:`GIMLI`, via
    :class:`ClusterResult`). This statelessness is what makes it safe
    to share a single algorithm instance across ``joblib`` worker
    processes.

    Class attributes
    -----------------
    name : str
        The identifier used as the key in :data:`MODEL_REGISTRY`
        (set this to the same string passed to
        ``@register_model("...")``).
    supports_probabilities : bool
        Whether :meth:`predict_proba` returns meaningful soft
        assignments (``False`` falls back to the default
        implementation, which returns ``None``).
    supports_model_selection : bool
        Whether :meth:`select_model` is implemented for this algorithm.
        :meth:`GIMLI.select_model` raises ``NotImplementedError`` if
        ``False``.
    supports_covariance : bool
        Whether this algorithm is parameterised by a covariance
        structure (e.g. GMM). When ``False``, ``covariance_type``
        arguments are accepted but ignored.

    Contract
    --------
    ``select_model(X, config) -> pd.DataFrame``
        Sweep ``config.cluster_range`` (and, if ``supports_covariance``,
        ``config.covariance_types``) and return one row per combination
        with at least a ``"n_components"`` column and every metric
        column the algorithm can compute (e.g. ``"bic"``, ``"aic"``).
        :meth:`GIMLI.select_model` requires the column named by
        ``config.selection`` to be present.

    ``fit(X, config, n_clusters, covariance_type=None) -> (model, ClusterMetrics)``
        Fit the final model with the given number of clusters (and
        covariance type, if applicable) and return the fitted estimator
        together with its :class:`ClusterMetrics`.

    ``predict(model, X) -> np.ndarray``
        Hard cluster labels for ``X`` under ``model``.

    ``predict_proba(model, X) -> np.ndarray or None``
        Soft membership probabilities, shape
        ``(n_samples, n_clusters)``. The default implementation
        returns ``None``; override when ``supports_probabilities = True``.
    """

    name = "base"

    supports_probabilities = False

    supports_model_selection = False

    supports_covariance = False

    @abstractmethod
    def select_model(self, X: np.ndarray, config: "ClusterConfig") -> pd.DataFrame:
        """Sweep candidate hyperparameters and return a metrics table."""
        ...

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        config: "ClusterConfig",
        n_clusters: int,
        covariance_type: Optional[str] = None,
    ) -> tuple[Any, "ClusterMetrics"]:
        """Fit the final model and return ``(model, metrics)``."""
        ...

    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return hard cluster labels for ``X`` under ``model``."""
        ...

    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Return soft membership probabilities for ``X`` under ``model``.

        The default implementation returns ``None``; subclasses with
        ``supports_probabilities = True`` must override this.
        """
        return None

    @staticmethod
    def compute_generic_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Compute the three model-agnostic geometric metrics shared by
        every clustering algorithm: silhouette score, Calinski-Harabasz
        index and Davies-Bouldin index.

        Centralising this computation here (rather than duplicating it
        in every :class:`BaseClusterModel` subclass) keeps each
        algorithm's ``fit`` implementation focused on what is specific
        to it.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape ``(n_samples, n_features)``.
        labels : np.ndarray
            Hard cluster labels, shape ``(n_samples,)``.

        Returns
        -------
        dict[str, float]
            Keys ``"silhouette"``, ``"calinski_harabasz"`` and
            ``"davies_bouldin"``. All three are ``NaN`` when fewer than
            two distinct labels are present, or when every sample
            shares one label (these metrics are undefined in that
            case).

        Examples
        --------
        ::

            metrics = BaseClusterModel.compute_generic_metrics(X, labels)
            print(metrics["silhouette"])
        """
        n_unique = len(np.unique(labels))
        if n_unique < 2 or n_unique >= len(X):
            return {
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
            }

        return {
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        }


@register_model("gmm")
class GMMModel(BaseClusterModel):
    """
    Gaussian Mixture Model implementation, wrapping
    ``sklearn.mixture.GaussianMixture``.

    Supports model selection (BIC/AIC sweep over both number of
    components and covariance type) and soft membership probabilities.
    """

    name = "gmm"

    supports_probabilities = True

    supports_model_selection = True

    supports_covariance = True

    def _build_estimator(
        self, n_components: int, covariance_type: str, config: "ClusterConfig"
    ) -> GaussianMixture:
        """Construct a configured (unfitted) GaussianMixture instance."""
        return GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=config.max_iter,
            tol=config.tol,
            n_init=config.n_init,
            random_state=config.random_state,
        )

    def select_model(self, X: np.ndarray, config: "ClusterConfig") -> pd.DataFrame:
        """
        Fit a ``GaussianMixture`` for every combination of
        ``config.cluster_range`` × ``config.covariance_types`` and
        record BIC, AIC, log-likelihood, convergence and iteration
        count for each.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix for a single group.
        config : ClusterConfig
            Active configuration (only ``cluster_range``,
            ``covariance_types``, ``max_iter``, ``tol``, ``n_init`` and
            ``random_state`` are used here).

        Returns
        -------
        pd.DataFrame
            Columns: ``n_components``, ``covariance_type``, ``bic``,
            ``aic``, ``log_likelihood``, ``converged``, ``n_iter``.
            A combination that fails to fit (e.g. too many components
            for too few samples with a ``"full"`` covariance) gets
            ``NaN`` metrics and ``converged=False`` rather than raising.

        Examples
        --------
        ::

            table = GMMModel().select_model(X, config)
            best = table.loc[table["bic"].idxmin()]
        """
        records = []
        for covariance_type in config.covariance_types:
            for n_components in config.cluster_range:
                model = self._build_estimator(n_components, covariance_type, config)
                try:
                    model.fit(X)
                    bic = model.bic(X)
                    aic = model.aic(X)
                    log_likelihood = model.score(X) * X.shape[0]
                    converged = bool(model.converged_)
                    n_iter = int(model.n_iter_)
                except Exception:
                    bic = aic = log_likelihood = np.nan
                    converged = False
                    n_iter = np.nan

                records.append(
                    {
                        "n_components": n_components,
                        "covariance_type": covariance_type,
                        "bic": bic,
                        "aic": aic,
                        "log_likelihood": log_likelihood,
                        "converged": converged,
                        "n_iter": n_iter,
                    }
                )

        return pd.DataFrame.from_records(records)

    def fit(
        self,
        X: np.ndarray,
        config: "ClusterConfig",
        n_clusters: int,
        covariance_type: Optional[str] = None,
    ) -> tuple[GaussianMixture, ClusterMetrics]:
        """
        Fit the final ``GaussianMixture`` and compute its full set of
        metrics (BIC, AIC, log-likelihood, mean assignment entropy,
        plus the model-agnostic silhouette / Calinski-Harabasz /
        Davies-Bouldin trio).

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix for a single group.
        config : ClusterConfig
            Active configuration.
        n_clusters : int
            Number of mixture components.
        covariance_type : str or None
            One of ``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``.
            Falls back to ``config.covariance_types[0]`` if ``None``.

        Returns
        -------
        tuple[GaussianMixture, ClusterMetrics]
            The fitted estimator and its metrics.

        Examples
        --------
        ::

            model, metrics = GMMModel().fit(X, config, n_clusters=3, covariance_type="diag")
            print(metrics.bic, metrics.silhouette)
        """
        cov_type = covariance_type or config.covariance_types[0]
        model = self._build_estimator(n_clusters, cov_type, config)
        model.fit(X)

        labels = model.predict(X)
        proba = model.predict_proba(X)
        safe_proba = np.clip(proba, 1e-12, 1.0)
        entropy = float(-np.mean(np.sum(safe_proba * np.log(safe_proba), axis=1)))

        generic = self.compute_generic_metrics(X, labels)

        metrics = ClusterMetrics(
            bic=model.bic(X),
            aic=model.aic(X),
            log_likelihood=model.score(X) * X.shape[0],
            entropy=entropy,
            converged=bool(model.converged_),
            iterations=int(model.n_iter_),
            silhouette=generic["silhouette"],
            calinski_harabasz=generic["calinski_harabasz"],
            davies_bouldin=generic["davies_bouldin"],
        )

        return model, metrics

    def predict(self, model: GaussianMixture, X: np.ndarray) -> np.ndarray:
        """Return the hard cluster assignment (argmax component) for ``X``."""
        return model.predict(X)

    def predict_proba(self, model: GaussianMixture, X: np.ndarray) -> np.ndarray:
        """Return the per-component posterior responsibilities for ``X``."""
        return model.predict_proba(X)


# ============================================================================
# Plotting
# ============================================================================

class ClusterPlotter:
    """
    Plotting namespace attached to every :class:`GIMLI` instance as
    ``gimli.plot``.

    Every method uses matplotlib's object-oriented API exclusively
    (``fig, ax = plt.subplots(...)``) rather than the stateful
    ``pyplot`` shortcuts (``plt.scatter`` and friends), so that two
    plots built back-to-back never collide on the same implicit
    "current axes" — a real bug present in the previous
    ``SAM.Weapons.GMM()`` implementation.

    Every method accepts an optional ``ax`` to draw into an
    already-existing axes (e.g. for composing a custom multi-panel
    figure) and an optional ``save_path`` to write the figure to disk
    immediately; both default to "create a new figure and just show
    it inline", which is what most interactive notebook use looks
    like.
    """

    def __init__(self, gimli: "GIMLI"):
        self.parent = gimli

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_group(self, group) -> tuple:
        """
        Normalise ``group`` (a scalar, a tuple, or ``None``) into the
        exact tuple key used in ``gimli.results``.

        If ``group`` is ``None``, this only succeeds when there is
        exactly one group in total (typically the ungrouped case,
        where the only key is ``()``); otherwise it raises asking for
        an explicit group.
        """
        gimli = self.parent
        if group is not None:
            key = group if isinstance(group, tuple) else (group,)
            if key not in gimli.results:
                raise KeyError(
                    f"Group {group!r} not found. "
                    f"Available groups: {list(gimli.results.keys())}."
                )
            return key

        if len(gimli.results) == 1:
            return next(iter(gimli.results.keys()))

        raise ValueError(
            "Multiple groups are available; please specify `group` explicitly. "
            f"Available groups: {list(gimli.results.keys())}."
        )

    def _metric_curve(
        self,
        metric: str,
        group=None,
        covariance_type: Optional[str] = None,
        ax=None,
        save_path: Optional[str] = None,
    ):
        """Shared implementation behind :meth:`bic` and :meth:`aic`."""
        gimli = self.parent
        gimli._require_prepared()

        groups = [self._resolve_group(group)] if group is not None else list(gimli.results.keys())

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        any_plotted = False
        for key in groups:
            result = gimli.results.get(key)
            if result is None or result.selection_table is None:
                continue

            table = result.selection_table
            if covariance_type is not None:
                table = table[table["covariance_type"] == covariance_type]

            multi_cov = table["covariance_type"].nunique() > 1 if "covariance_type" in table.columns else False

            group_iter = (
                table.groupby("covariance_type")
                if "covariance_type" in table.columns
                else [(None, table)]
            )
            for cov, sub in group_iter:
                sub = sub.sort_values("n_components")
                if not len(sub):
                    continue
                label_parts = []
                if len(groups) > 1:
                    label_parts.append(str(key))
                if multi_cov and cov is not None:
                    label_parts.append(str(cov))
                label = " / ".join(label_parts) if label_parts else metric.upper()
                ax.plot(sub["n_components"], sub[metric], marker="o", label=label)
                any_plotted = True

        ax.set_xlabel("Number of components")
        ax.set_ylabel(metric.upper())
        title_group = str(groups[0]) if len(groups) == 1 else f"{len(groups)} groups"
        ax.set_title(f"{metric.upper()} vs number of components — {title_group}")
        ax.grid(True, linestyle="--", alpha=0.4)
        if any_plotted and (len(groups) > 1 or covariance_type is None):
            ax.legend(fontsize=8)

        if fig is not None:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return ax

    # ------------------------------------------------------------------
    # Public plots
    # ------------------------------------------------------------------

    def bic(
        self,
        group=None,
        covariance_type: Optional[str] = None,
        ax=None,
        save_path: Optional[str] = None,
    ):
        """
        Plot BIC versus number of components for one or every group.

        Parameters
        ----------
        group : tuple, scalar or None
            Which group to plot. If ``None``, every group with a
            selection table is overlaid on the same axes (one line per
            group, or per group/covariance-type combination).
        covariance_type : str or None
            Restrict the plot to a single covariance type. If ``None``
            and a group's selection table contains several, one line
            per covariance type is drawn.
        ax : matplotlib.axes.Axes or None
            Draw into this axes instead of creating a new figure.
        save_path : str or None
            If given, save the figure to this path.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        ::

            cluster.plot.bic(group=(2.0, 0.7))
            cluster.plot.bic()  # overlay every group
        """
        return self._metric_curve("bic", group=group, covariance_type=covariance_type, ax=ax, save_path=save_path)

    def aic(
        self,
        group=None,
        covariance_type: Optional[str] = None,
        ax=None,
        save_path: Optional[str] = None,
    ):
        """
        Plot AIC versus number of components. See :meth:`bic` for the
        full parameter description; the only difference is the metric
        plotted.

        Examples
        --------
        ::

            cluster.plot.aic(group=(2.0, 0.7))
        """
        return self._metric_curve("aic", group=group, covariance_type=covariance_type, ax=ax, save_path=save_path)

    def scatter(self, group=None, ax=None, save_path: Optional[str] = None):
        """
        Plot a 2-D scatter of one group's points, coloured by cluster
        label.

        When the group has more than two features, a PCA projection to
        two dimensions is used automatically and the axis labels
        report the explained variance ratio of each component.

        Parameters
        ----------
        group : tuple, scalar or None
            Which group to plot. Required unless there is exactly one
            group in total.
        ax : matplotlib.axes.Axes or None
            Draw into this axes instead of creating a new figure.
        save_path : str or None
            If given, save the figure to this path.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If the group has not been fitted yet.

        Examples
        --------
        ::

            cluster.prepare().select_model().fit().predict()
            cluster.plot.scatter(group=(2.0, 0.7))
        """
        gimli = self.parent
        gimli._require_fitted()

        key = self._resolve_group(group)
        result = gimli.results.get(key)
        if result is None or result.model is None:
            raise RuntimeError(f"Group {key} has not been fitted.")

        idx = gimli.dataset.group_indices[key]
        X = gimli.dataset.X_scaled[idx]
        labels = result.labels if result.labels is not None else gimli._algorithm.predict(result.model, X)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        feature_names = gimli.dataset.feature_names
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=gimli.config.random_state)
            proj = pca.fit_transform(X)
            var = pca.explained_variance_ratio_
            xlabel = f"PC1 ({var[0] * 100:.1f}% var)"
            ylabel = f"PC2 ({var[1] * 100:.1f}% var)"
        elif X.shape[1] == 2:
            proj = X
            xlabel, ylabel = feature_names[0], feature_names[1]
        else:
            proj = np.column_stack([X[:, 0], np.zeros(len(X))])
            xlabel, ylabel = feature_names[0], ""

        sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="viridis", s=30, edgecolor="k")
        (fig if fig is not None else ax.figure).colorbar(sc, ax=ax, label="Cluster")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Clusters — group {key}" if key else "Clusters")
        ax.grid(True, linestyle="--", alpha=0.3)

        if fig is not None:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return ax

    def heatmap(
        self,
        value: str = "recommended_n",
        ax=None,
        save_path: Optional[str] = None,
        annot: Optional[bool] = None,
        cmap: str = "viridis",
    ):
        """
        Plot a heatmap of a summary metric across the two grouping
        columns.

        Requires exactly two grouping columns (``len(gimli.groups) == 2``),
        since the heatmap's two axes are precisely those two columns.
        Builds (or reuses) :attr:`GIMLI.summary_table` internally.

        Parameters
        ----------
        value : str
            Column of the summary table to plot — any of
            ``"recommended_n"``, ``"n_clusters_used"``, ``"bic"``,
            ``"silhouette"``, etc. Default ``"recommended_n"``.
        ax : matplotlib.axes.Axes or None
            Draw into this axes instead of creating a new figure.
        save_path : str or None
            If given, save the figure to this path.
        annot : bool or None
            Whether to annotate each cell with its value. Defaults to
            ``True`` only when the grid has at most 225 cells (15x15),
            to keep larger heatmaps readable.
        cmap : str
            Matplotlib colormap name. Default ``"viridis"``.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        ValueError
            If ``gimli.groups`` does not have exactly two columns, or
            ``value`` is not a column of the summary table.

        Examples
        --------
        ::

            cluster.plot.heatmap(value="recommended_n")
            cluster.plot.heatmap(value="silhouette", cmap="coolwarm")
        """
        gimli = self.parent
        if len(gimli.groups) != 2:
            raise ValueError(
                "heatmap() requires exactly two grouping columns "
                f"(got {gimli.groups})."
            )

        if gimli.summary_table is None:
            gimli.summary()

        table = gimli.summary_table
        if value not in table.columns:
            raise ValueError(
                f"'{value}' not found in the summary table. "
                f"Available columns: {list(table.columns)}."
            )

        pivot = table.pivot(index=gimli.groups[0], columns=gimli.groups[1], values=value)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 7))

        if annot is None:
            annot = pivot.size <= 225

        sns.heatmap(pivot, annot=annot, fmt=".2f", cmap=cmap, ax=ax, cbar=True)
        ax.set_title(f"{value} across {gimli.groups[0]} × {gimli.groups[1]}")
        ax.set_xlabel(gimli.groups[1])
        ax.set_ylabel(gimli.groups[0])

        if fig is not None:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return ax

    def probabilities(self, group=None, ax=None, save_path: Optional[str] = None):
        """
        Plot a boxplot of the maximum membership probability per
        cluster, for one group — a diagnostic for how confidently
        points are assigned (low values indicate overlapping clusters).

        Parameters
        ----------
        group : tuple, scalar or None
            Which group to plot. Required unless there is exactly one
            group in total.
        ax : matplotlib.axes.Axes or None
            Draw into this axes instead of creating a new figure.
        save_path : str or None
            If given, save the figure to this path.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If :meth:`GIMLI.predict` has not been called, or the
            algorithm does not support soft assignments.

        Examples
        --------
        ::

            cluster.prepare().select_model().fit().predict()
            cluster.plot.probabilities(group=(2.0, 0.7))
        """
        gimli = self.parent
        gimli._require_fitted()

        key = self._resolve_group(group)
        result = gimli.results.get(key)
        if result is None or result.probabilities is None:
            raise RuntimeError(
                f"No membership probabilities available for group {key}. "
                "Either the algorithm does not support soft assignments, "
                "or predict() has not been called yet."
            )

        max_proba = result.probabilities.max(axis=1)
        labels = result.labels

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        unique_labels = np.unique(labels)
        data = [max_proba[labels == c] for c in unique_labels]
        ax.boxplot(data, tick_labels=[str(c) for c in unique_labels])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Max membership probability")
        ax.set_title(f"Assignment confidence — group {key}" if key else "Assignment confidence")
        ax.grid(True, linestyle="--", alpha=0.3)

        if fig is not None:
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return ax