#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Logistic Regression Pipeline for Microbiome-Based Health Classification

Overview
--------
This script implements an optimized logistic regression workflow for a custom
microbiome-based health classification task.

Key design choices
------------------
1. Preprocessing is fitted once per CV fold and reused across all C values.
2. The script compares:
   - L1 logistic regression
   - L1 feature selection followed by L2 refit
   - Linear combinations of the two decision scores
3. The final classification threshold is selected from repeated-CV
   out-of-fold (OOF) decision scores.
4. Cross-validation and final training use the same preprocessing and
   modeling logic.
5. Continuous residual features and binary presence features are filtered
   independently.
6. Missing residual values in external data are imputed using the training
   medians instead of arbitrary constants.
7. The script exports:
   - trained model bundle
   - coefficients
   - OOF predictions
   - feature selection frequency
   - external validation summaries

Important notes
---------------
- This is a custom health classification model. It is NOT an official or
  validated GMWI2 implementation.
- CV tuning metrics are used for model selection and should not be reported
  as independent generalization performance.
- Generalization performance should be judged using held-out validation and
  external datasets.
"""

import os
import json
import time
import inspect
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# ============================================================================
# Configuration
# ============================================================================

BACTERIA_CSV = "bacteria_residual.csv"
RAW_COUNT_CSV = "bracken_bacteria_species_1.csv"
META_CSV = "all_metadata.csv"

OUT_DIR = "lasso25"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEEDS = [42]

CLASS_WEIGHT = "balanced"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.3

N_SPLITS = 5
N_REPEATS = 3

PRESENCE_THRESHOLD = 0.0
PREVALENCE_THRESHOLD = 0.01

# Variance threshold for log1p-transformed raw counts
RAW_LOG_VAR_THRESHOLD = 1e-6

# Variance threshold for scaled residual features
CONT_VAR_THRESHOLD = 1e-6

# Variance threshold for binary presence features
BIN_VAR_THRESHOLD = 1e-6

ENABLE_WINSORIZE = True
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99

ENABLE_CORR_CLUSTER = True
CORR_THRESHOLD = 0.95
CORR_SAMPLE_SIZE = 2000

ROBUST_QUANTILE_RANGE = (10.0, 90.0)

C_GRID = np.unique(
    np.concatenate([
        np.logspace(-3.5, 0.7, 22),
        [0.03, 0.05, 0.07, 0.12, 0.2, 0.3],
    ])
)

ENABLE_RELAXED_REFIT = True
REFIT_L2_C = 1.0

# Refit is only attempted when the L1 support size is within this limit.
MAX_REFIT_FEATURES = 300

# Final score = (1 - alpha) * L1 score + alpha * refit score
RELAX_ALPHAS = [0.0, 0.5, 1.0]

COEF_EPS = 1e-10
LR_MAX_ITER = 2000
LR_TOL = 1e-4

# Allowed values:
# - "auc"
# - "balanced_accuracy"
# - "combined"
MODEL_SELECTION_METRIC = "auc"

N_BOOTSTRAP = 2000

N_ANNOTATE_POS = 7
N_ANNOTATE_NEG = 7

# Three-way prediction based on centered decision score:
#   decision_score = raw_score - tuned_score_threshold
#   decision_score >  c  -> healthy
#   decision_score < -c  -> non-healthy
#   otherwise            -> defer
CUTOFF_C = 0.0

# If an external raw-taxon column is entirely missing:
# True  -> fill with 0, interpreted as not detected
# False -> treat as unknown and impute presence using training prevalence
EXTERNAL_MISSING_RAW_IS_ZERO = True

EXTERNAL_DATASETS = [
    {
        "path": "D:/lasso/Daiki_Ventura",
        "tag": "Daiki_Ventura",
        "residual": "bacteria_residual.csv",
        "raw": "bracken_bacteria_species.csv",
        "metadata": "metadata.csv",
    },
    {
        "path": "D:/lasso/AS1",
        "tag": "AS1",
        "residual": "bacteria_residual.csv",
        "raw": "bracken_bacteria_species.csv",
        "metadata": "metadata.csv",
    },
    {
        "path": "D:/lasso/AS2",
        "tag": "AS2",
        "residual": "bacteria_residual.csv",
        "raw": "bracken_bacteria_species.csv",
        "metadata": "metadata.csv",
    },
    {
        "path": "D:/lasso/Daiki/NMOSD",
        "tag": "NMOSD",
        "residual": "bacteria_residual.csv",
        "raw": "bracken_bacteria_species.csv",
        "metadata": "metadata.csv",
    },
]

# ============================================================================
# Plot style
# ============================================================================


def set_publication_style():
    """Apply a clean publication-style Matplotlib/Seaborn theme."""
    sns.set_style("whitegrid")

    mpl.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "figure.figsize": (7.0, 5.5),
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "axes.linewidth": 1.1,
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "legend.fontsize": 10,
        "legend.frameon": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


set_publication_style()

# ============================================================================
# Data loading
# ============================================================================


def read_feature_matrix(path):
    """
    Read a taxa-by-sample CSV file and return a sample-by-feature DataFrame.

    The input file is expected to have taxa as rows and samples as columns.
    This function transposes the matrix so that rows are samples and columns
    are taxa.

    Parameters
    ----------
    path : str
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Numeric DataFrame indexed by SampleID with taxa columns.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(path, index_col=0).T

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    if df.index.duplicated().any():
        bad = df.index[df.index.duplicated()].tolist()[:10]
        raise ValueError(f"Duplicate SampleID in {path}: {bad}")

    if df.columns.duplicated().any():
        bad = df.columns[df.columns.duplicated()].tolist()[:10]
        raise ValueError(f"Duplicate taxa in {path}: {bad}")

    df = df.apply(pd.to_numeric, errors="raise").astype(np.float64)
    return df.replace([np.inf, -np.inf], np.nan)


def load_dataset(residual_path, raw_path, metadata_path, external=False):
    """
    Load residual features, raw count features, and metadata.

    Parameters
    ----------
    residual_path : str
        Path to the residual feature matrix.
    raw_path : str
        Path to the raw count feature matrix.
    metadata_path : str
        Path to the metadata table.
    external : bool, default=False
        Whether the dataset is external. Internal datasets require overlap
        between residual and raw taxa before training.

    Returns
    -------
    tuple
        (X_cont, X_raw, y, sample_ids)
    """
    X_cont = read_feature_matrix(residual_path)
    X_raw = read_feature_matrix(raw_path)

    meta = pd.read_csv(metadata_path, dtype={"SampleID": str})

    required = {"SampleID", "MS"}
    if not required.issubset(meta.columns):
        raise ValueError(f"{metadata_path} must contain columns: SampleID, MS")

    if meta["SampleID"].isna().any():
        raise ValueError(f"Missing SampleID in {metadata_path}")

    meta["SampleID"] = meta["SampleID"].str.strip()

    if meta["SampleID"].duplicated().any():
        raise ValueError(f"Duplicate SampleID in {metadata_path}")

    common_ids = (
        X_cont.index
        .intersection(X_raw.index, sort=False)
        .intersection(pd.Index(meta["SampleID"]), sort=False)
    )

    if len(common_ids) == 0:
        raise ValueError("No overlapping SampleID among the three files.")

    n_before = (len(X_cont), len(X_raw), len(meta))

    X_cont = X_cont.loc[common_ids].copy()
    X_raw = X_raw.loc[common_ids].copy()
    meta = meta.set_index("SampleID").loc[common_ids].copy()

    ms = pd.to_numeric(meta["MS"], errors="raise")

    if ms.isna().any() or not ms.isin([0, 1]).all():
        raise ValueError("MS must contain only 0 and 1, without missing values.")

    # Positive class is defined as healthy = 1 - MS
    y = (1 - ms.astype(int)).to_numpy(dtype=np.int64)

    if not external:
        common_taxa = X_cont.columns.intersection(X_raw.columns, sort=False)

        if len(common_taxa) == 0:
            raise ValueError("No overlapping taxa in residual and raw matrices.")

        X_cont = X_cont.loc[:, common_taxa].copy()
        X_raw = X_raw.loc[:, common_taxa].copy()

        if X_raw.isna().any().any():
            raise ValueError(
                "Internal raw count matrix contains NaN/Inf. "
                "Please resolve missing raw counts before training."
            )

        if np.unique(y).size != 2:
            raise ValueError("Internal dataset must contain both classes.")

    if np.any(X_raw.to_numpy() < 0):
        raise ValueError("Raw count matrix contains negative values.")

    print(
        f"Loaded N={len(y)}; "
        f"healthy={int((y == 1).sum())}; "
        f"non-healthy={int((y == 0).sum())}; "
        f"residual taxa={X_cont.shape[1]}; "
        f"raw taxa={X_raw.shape[1]}"
    )

    if any(n != len(common_ids) for n in n_before):
        print(
            f"  Sample intersection: original residual/raw/meta={n_before}, "
            f"retained={len(common_ids)}"
        )

    return X_cont, X_raw, y, common_ids.to_numpy()

# ============================================================================
# Preprocessing
# ============================================================================


def correlation_cluster_indices(X, corr_threshold, seed):
    """
    Select one representative feature from each high-correlation cluster.

    Features are clustered using:
        distance = 1 - abs(correlation)

    Within each cluster, the feature with the largest variance is retained.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n_samples, n_features).
    corr_threshold : float
        Absolute correlation threshold used to define clusters.
    seed : int
        Random seed for optional row subsampling.

    Returns
    -------
    np.ndarray
        Sorted indices of representative features.
    """
    n, p = X.shape

    if p <= 1:
        return np.arange(p, dtype=int)

    if n > CORR_SAMPLE_SIZE:
        rng = np.random.default_rng(seed)
        rows = rng.choice(n, CORR_SAMPLE_SIZE, replace=False)
        A = X[rows]
    else:
        A = X

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(A, rowvar=False)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    dist = np.clip(1.0 - np.abs(corr), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    clusters = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")
    variances = np.var(A, axis=0)

    representatives = []
    for cid in np.unique(clusters):
        idx = np.flatnonzero(clusters == cid)
        representatives.append(idx[np.argmax(variances[idx])])

    return np.sort(np.asarray(representatives, dtype=int))


def fit_preprocessor(X_cont, X_raw, seed):
    """
    Fit fold-specific preprocessing on training data only.

    Steps
    -----
    1. Filter taxa by prevalence using raw counts.
    2. Impute missing residual values using training medians.
    3. Optionally winsorize residual features.
    4. Robust-scale residual features.
    5. Filter residual features using raw-count variance and scaled-residual variance.
    6. Optionally perform correlation clustering on residual features.
    7. Independently retain binary presence features.

    Parameters
    ----------
    X_cont : pd.DataFrame
        Residual feature matrix.
    X_raw : pd.DataFrame
        Raw count feature matrix.
    seed : int
        Random seed used in correlation clustering.

    Returns
    -------
    dict
        Fitted preprocessing state.
    """
    taxa = list(X_cont.columns)

    raw = X_raw.loc[X_cont.index, taxa].to_numpy(dtype=np.float64)
    cont = X_cont.to_numpy(dtype=np.float64)

    presence = raw > PRESENCE_THRESHOLD
    prevalence = presence.mean(axis=0)

    prev_idx = np.flatnonzero(prevalence >= PREVALENCE_THRESHOLD)
    if prev_idx.size == 0:
        raise ValueError("All taxa removed by prevalence filtering.")

    prev_taxa = [taxa[i] for i in prev_idx]

    raw_prev = raw[:, prev_idx]
    cont_prev = cont[:, prev_idx]
    prevalence_prev = prevalence[prev_idx]

    # Impute residual values using training medians.
    # Columns that are entirely missing are temporarily filled with 0,
    # and will usually be removed later by variance filtering.
    median_series = pd.DataFrame(cont_prev).median(axis=0).fillna(0.0)
    medians = median_series.to_numpy(dtype=np.float64)

    cont_filled = np.where(np.isfinite(cont_prev), cont_prev, medians[None, :])

    if ENABLE_WINSORIZE:
        lower = np.quantile(cont_filled, WINSOR_LOWER_Q, axis=0)
        upper = np.quantile(cont_filled, WINSOR_UPPER_Q, axis=0)
        cont_clipped = np.clip(cont_filled, lower, upper)
    else:
        lower = None
        upper = None
        cont_clipped = cont_filled

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=ROBUST_QUANTILE_RANGE,
    )
    cont_scaled = scaler.fit_transform(cont_clipped)

    # Variance filters are applied on meaningful scales:
    # - raw counts on log1p scale
    # - residual features after robust scaling
    raw_log_var = np.var(np.log1p(raw_prev), axis=0)
    cont_var = np.var(cont_scaled, axis=0)

    cont_keep = np.flatnonzero(
        (raw_log_var > RAW_LOG_VAR_THRESHOLD) &
        (cont_var > CONT_VAR_THRESHOLD)
    )

    if ENABLE_CORR_CLUSTER and cont_keep.size > 1:
        local_rep = correlation_cluster_indices(
            cont_scaled[:, cont_keep],
            corr_threshold=CORR_THRESHOLD,
            seed=seed,
        )
        cont_keep = cont_keep[local_rep]

    # Binary features are filtered independently and are not removed by
    # residual correlation clustering.
    bin_var = prevalence_prev * (1.0 - prevalence_prev)
    bin_keep = np.flatnonzero(bin_var > BIN_VAR_THRESHOLD)

    cont_names = [prev_taxa[i] for i in cont_keep]
    bin_taxa = [prev_taxa[i] for i in bin_keep]

    feature_names = (
        [f"{t}__residual" for t in cont_names] +
        [f"{t}__bin" for t in bin_taxa]
    )

    feature_taxa = cont_names + bin_taxa
    feature_types = (
        ["continuous_residual"] * len(cont_names) +
        ["binary_presence"] * len(bin_taxa)
    )

    if not feature_names:
        raise ValueError("No features remained after preprocessing.")

    return {
        "taxa_all": taxa,
        "prev_taxa": prev_taxa,
        "medians": medians,
        "prevalence": prevalence_prev,
        "lower": lower,
        "upper": upper,
        "scaler": scaler,
        "cont_keep": cont_keep,
        "bin_keep": bin_keep,
        "feature_names": feature_names,
        "feature_taxa": feature_taxa,
        "feature_types": feature_types,
        "presence_threshold": PRESENCE_THRESHOLD,
    }


def transform_with_preprocessor(prep, X_cont, X_raw):
    """
    Apply a fitted preprocessor to new data.

    Parameters
    ----------
    prep : dict
        Fitted preprocessing state.
    X_cont : pd.DataFrame
        Residual feature matrix.
    X_raw : pd.DataFrame
        Raw count feature matrix.

    Returns
    -------
    np.ndarray
        Final processed design matrix.
    """
    taxa = prep["prev_taxa"]

    cont = X_cont.reindex(columns=taxa).to_numpy(dtype=np.float64)
    raw = X_raw.reindex(index=X_cont.index, columns=taxa).to_numpy(dtype=np.float64)

    cont = np.where(np.isfinite(cont), cont, prep["medians"][None, :])

    if prep["lower"] is not None:
        cont = np.clip(cont, prep["lower"], prep["upper"])

    cont_scaled = prep["scaler"].transform(cont)
    cont_final = cont_scaled[:, prep["cont_keep"]]

    # Missing raw entries are treated as unknown presence status.
    # Known raw=0 remains strict absence=0.
    known = np.isfinite(raw)
    binary = (raw > prep["presence_threshold"]).astype(np.float64)
    binary = np.where(known, binary, prep["prevalence"][None, :])
    bin_final = binary[:, prep["bin_keep"]]

    X = np.hstack([cont_final, bin_final])

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values remained after preprocessing.")

    return np.ascontiguousarray(X, dtype=np.float64)

# ============================================================================
# Logistic regression utilities
# ============================================================================


def make_logistic_regression(C, penalty, seed):
    """
    Create a LogisticRegression estimator configured for this pipeline.

    Parameters
    ----------
    C : float
        Inverse regularization strength.
    penalty : str
        Penalty type, either 'l1' or 'l2'.
    seed : int
        Random seed.

    Returns
    -------
    LogisticRegression
        Configured estimator.
    """
    kwargs = {
        "solver": "liblinear",
        "class_weight": CLASS_WEIGHT,
        "random_state": int(seed),
        "C": float(C),
        "max_iter": LR_MAX_ITER,
        "tol": LR_TOL,
        "fit_intercept": True,
    }

    params = inspect.signature(LogisticRegression).parameters
    penalty_param = params.get("penalty")

    if "l1_ratio" in params and (penalty_param is None or penalty_param.default == "deprecated"):
        kwargs["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        kwargs["penalty"] = penalty

    return LogisticRegression(**kwargs)


def fit_l1_and_optional_refit(X, y, C, seed, refit_cache=None):
    """
    Fit L1 logistic regression and optionally refit an L2 model on the L1 support.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.
    y : np.ndarray
        Binary labels.
    C : float
        Inverse regularization strength for the L1 model.
    seed : int
        Random seed.
    refit_cache : dict, optional
        Fold-local cache keyed by L1 support set.

    Returns
    -------
    dict
        Dictionary containing L1 coefficients, optional refit coefficients,
        support indices, and a flag indicating whether refit was used.
    """
    lr_l1 = make_logistic_regression(C, penalty="l1", seed=seed)
    lr_l1.fit(X, y)

    w_l1 = lr_l1.coef_.ravel().copy()
    b_l1 = float(lr_l1.intercept_[0])

    support = np.flatnonzero(np.abs(w_l1) > COEF_EPS)

    w_refit = w_l1.copy()
    b_refit = b_l1
    refit_used = False

    can_refit = ENABLE_RELAXED_REFIT and 1 <= support.size <= MAX_REFIT_FEATURES

    if can_refit:
        key = tuple(support.tolist())

        if refit_cache is not None and key in refit_cache:
            selected_coef, selected_intercept = refit_cache[key]
        else:
            lr_l2 = make_logistic_regression(REFIT_L2_C, penalty="l2", seed=seed)
            lr_l2.fit(X[:, support], y)

            selected_coef = lr_l2.coef_.ravel().copy()
            selected_intercept = float(lr_l2.intercept_[0])

            if refit_cache is not None:
                refit_cache[key] = (selected_coef.copy(), selected_intercept)

        w_refit = np.zeros(X.shape[1], dtype=np.float64)
        w_refit[support] = selected_coef
        b_refit = selected_intercept
        refit_used = True

    return {
        "w_l1": w_l1,
        "b_l1": b_l1,
        "w_refit": w_refit,
        "b_refit": b_refit,
        "support": support,
        "refit_used": refit_used,
    }


def combine_coefficients(parts, alpha):
    """
    Combine L1 and optional refit coefficients by linear interpolation.

    Parameters
    ----------
    parts : dict
        Output from fit_l1_and_optional_refit.
    alpha : float
        Mixing weight in [0, 1].

    Returns
    -------
    tuple
        (combined_coef, combined_intercept)
    """
    alpha = float(alpha)

    w = (1.0 - alpha) * parts["w_l1"] + alpha * parts["w_refit"]
    b = (1.0 - alpha) * parts["b_l1"] + alpha * parts["b_refit"]

    return w, float(b)

# ============================================================================
# Threshold tuning
# ============================================================================


def find_best_score_threshold(y, scores):
    """
    Find the decision-score threshold that maximizes balanced accuracy.

    Thresholds are evaluated at the midpoints between sorted unique scores,
    plus sentinel values around the extremes. Ties are broken by selecting
    the threshold closest to zero.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    scores : np.ndarray
        Decision scores.

    Returns
    -------
    tuple
        (best_threshold, best_balanced_accuracy)
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Threshold tuning requires both classes.")

    unique_scores = np.unique(scores)

    if unique_scores.size > 1:
        middle = unique_scores[:-1] + (unique_scores[1:] - unique_scores[:-1]) / 2.0
    else:
        middle = np.empty(0, dtype=np.float64)

    thresholds = np.unique(np.concatenate([
        [
            np.nextafter(unique_scores[0], -np.inf),
            0.0,
            np.nextafter(unique_scores[-1], np.inf),
        ],
        middle,
    ]))

    # Prediction rule:
    # score >= threshold -> healthy
    fn = np.searchsorted(pos, thresholds, side="left")
    tn = np.searchsorted(neg, thresholds, side="left")

    tpr = 1.0 - fn / len(pos)
    tnr = tn / len(neg)
    balacc = 0.5 * (tpr + tnr)

    best_value = float(np.max(balacc))
    ties = np.flatnonzero(np.isclose(balacc, best_value, rtol=0.0, atol=1e-12))
    best_idx = ties[np.argmin(np.abs(thresholds[ties]))]

    return float(thresholds[best_idx]), best_value

# ============================================================================
# Cross-validation model selection
# ============================================================================


def select_model_by_cv(X_cont, X_raw, y, seed, out_dir):
    """
    Select the best model using repeated stratified cross-validation.

    Key optimization:
    Preprocessing is fitted once per fold and then reused for all C values.

    Parameters
    ----------
    X_cont : pd.DataFrame
        Residual feature matrix for the training split.
    X_raw : pd.DataFrame
        Raw count feature matrix for the training split.
    y : np.ndarray
        Binary labels.
    seed : int
        Random seed.
    out_dir : str
        Output directory.

    Returns
    -------
    tuple
        (best_candidate, best_oof_scores, cv_table, stability_table)
    """
    min_class_size = int(np.bincount(y, minlength=2).min())
    actual_splits = min(N_SPLITS, min_class_size)

    if actual_splits < 2:
        raise ValueError("Too few samples in the minority class for cross-validation.")

    if actual_splits != N_SPLITS:
        warnings.warn(
            f"N_SPLITS reduced from {N_SPLITS} to {actual_splits} "
            f"because of minority-class sample size."
        )

    alphas = sorted(set(float(a) for a in RELAX_ALPHAS)) if ENABLE_RELAXED_REFIT else [0.0]

    if any(a < 0.0 or a > 1.0 for a in alphas):
        raise ValueError("RELAX_ALPHAS must lie within [0, 1].")

    candidates = [
        {"c_index": ci, "C": float(C), "alpha": alpha}
        for ci, C in enumerate(C_GRID)
        for alpha in alphas
    ]

    n_candidates = len(candidates)
    n_samples = len(y)
    n_folds = actual_splits * N_REPEATS

    oof_score_sum = np.zeros((n_candidates, n_samples), dtype=np.float64)
    oof_count = np.zeros(n_samples, dtype=np.int32)

    fold_auc = np.zeros((n_candidates, n_folds))
    fold_feature_count = np.zeros((n_candidates, n_folds))
    fold_refit_used = np.zeros((n_candidates, n_folds))

    l1_selection_counts = [Counter() for _ in C_GRID]
    feature_available_counts = Counter()

    cv = RepeatedStratifiedKFold(
        n_splits=actual_splits,
        n_repeats=N_REPEATS,
        random_state=seed,
    )

    candidate_ids_by_c = {
        ci: [idx for idx, candidate in enumerate(candidates) if candidate["c_index"] == ci]
        for ci in range(len(C_GRID))
    }

    print(
        f"\nCV: {actual_splits} folds × {N_REPEATS} repeats; "
        f"{len(C_GRID)} C values; {len(alphas)} alpha values."
    )

    start = time.perf_counter()

    for fold_idx, (tr, te) in enumerate(cv.split(X_cont, y)):
        fold_start = time.perf_counter()

        Xc_tr = X_cont.iloc[tr]
        Xc_te = X_cont.iloc[te]
        Xr_tr = X_raw.iloc[tr]
        Xr_te = X_raw.iloc[te]
        y_tr = y[tr]
        y_te = y[te]

        prep = fit_preprocessor(Xc_tr, Xr_tr, seed=seed + fold_idx)
        X_tr = transform_with_preprocessor(prep, Xc_tr, Xr_tr)
        X_te = transform_with_preprocessor(prep, Xc_te, Xr_te)

        feature_names = prep["feature_names"]
        feature_available_counts.update(feature_names)

        refit_cache = {}

        for ci, C in enumerate(C_GRID):
            parts = fit_l1_and_optional_refit(
                X_tr,
                y_tr,
                C=float(C),
                seed=seed + fold_idx,
                refit_cache=refit_cache,
            )

            l1_selection_counts[ci].update(feature_names[i] for i in parts["support"])

            score_l1 = X_te @ parts["w_l1"] + parts["b_l1"]
            score_refit = X_te @ parts["w_refit"] + parts["b_refit"] if parts["refit_used"] else score_l1

            for candidate_idx in candidate_ids_by_c[ci]:
                alpha = candidates[candidate_idx]["alpha"]
                scores = (1.0 - alpha) * score_l1 + alpha * score_refit

                oof_score_sum[candidate_idx, te] += scores
                fold_auc[candidate_idx, fold_idx] = roc_auc_score(y_te, scores)

                w, _ = combine_coefficients(parts, alpha)
                fold_feature_count[candidate_idx, fold_idx] = np.count_nonzero(np.abs(w) > COEF_EPS)
                fold_refit_used[candidate_idx, fold_idx] = int(alpha > 0.0 and parts["refit_used"])

        oof_count[te] += 1

        print(
            f"  Fold {fold_idx + 1:02d}/{n_folds}: "
            f"processed features={X_tr.shape[1]}, "
            f"time={time.perf_counter() - fold_start:.1f}s"
        )

        del X_tr, X_te, refit_cache

    if not np.all(oof_count == N_REPEATS):
        raise RuntimeError("Unexpected OOF prediction counts.")

    oof_scores = oof_score_sum / oof_count[None, :]

    rows = []
    for candidate_idx, candidate in enumerate(candidates):
        scores = oof_scores[candidate_idx]

        score_threshold, oof_bal = find_best_score_threshold(y, scores)
        mean_auc = float(np.mean(fold_auc[candidate_idx]))

        if MODEL_SELECTION_METRIC == "auc":
            selection_score = mean_auc
        elif MODEL_SELECTION_METRIC == "balanced_accuracy":
            selection_score = oof_bal
        elif MODEL_SELECTION_METRIC == "combined":
            selection_score = 0.5 * mean_auc + 0.5 * oof_bal
        else:
            raise ValueError(f"Unknown MODEL_SELECTION_METRIC: {MODEL_SELECTION_METRIC}")

        rows.append({
            "candidate_id": candidate_idx,
            "C": candidate["C"],
            "alpha": candidate["alpha"],
            "selection_score": selection_score,
            "mean_cv_auc": mean_auc,
            "std_cv_auc": float(np.std(fold_auc[candidate_idx], ddof=1)),
            "oof_auc": float(roc_auc_score(y, scores)),
            "oof_tuned_balanced_accuracy": oof_bal,
            "score_threshold": score_threshold,
            "probability_threshold": float(expit(score_threshold)),
            "mean_nonzero_features": float(np.mean(fold_feature_count[candidate_idx])),
            "refit_fold_fraction": float(np.mean(fold_refit_used[candidate_idx])),
        })

    cv_table = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "mean_cv_auc",
            "oof_tuned_balanced_accuracy",
            "mean_nonzero_features",
            "alpha",
            "C",
        ],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)

    cv_table.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)

    best_row = cv_table.iloc[0]
    best_id = int(best_row["candidate_id"])
    best_candidate = candidates[best_id].copy()
    best_candidate["score_threshold"] = float(best_row["score_threshold"])
    best_candidate["actual_cv_splits"] = actual_splits
    best_candidate["cv_repeats"] = N_REPEATS

    best_ci = best_candidate["c_index"]
    selected_counter = l1_selection_counts[best_ci]

    stability_rows = []
    for feature in sorted(feature_available_counts):
        available = feature_available_counts[feature]
        selected = selected_counter[feature]

        stability_rows.append({
            "feature": feature,
            "available_folds": available,
            "selected_folds": selected,
            "selection_frequency_all_folds": selected / n_folds,
            "selection_frequency_when_available": selected / available if available else np.nan,
        })

    stability = pd.DataFrame(stability_rows).sort_values(
        ["selection_frequency_all_folds", "selection_frequency_when_available"],
        ascending=False,
    )

    stability.to_csv(os.path.join(out_dir, "feature_selection_frequency.csv"), index=False)

    print("\nTop CV candidates:")
    display_columns = [
        "C",
        "alpha",
        "mean_cv_auc",
        "oof_tuned_balanced_accuracy",
        "mean_nonzero_features",
        "refit_fold_fraction",
    ]
    print(
        cv_table[display_columns].head(10).to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )
    print(f"CV time: {time.perf_counter() - start:.1f}s")

    return best_candidate, oof_scores[best_id].copy(), cv_table, stability


def fit_final_model(X_cont, X_raw, y, best, seed):
    """
    Fit the final model on the full internal training split.

    Parameters
    ----------
    X_cont : pd.DataFrame
        Residual feature matrix.
    X_raw : pd.DataFrame
        Raw count feature matrix.
    y : np.ndarray
        Binary labels.
    best : dict
        Best hyperparameter setting selected by CV.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Final model bundle containing preprocessing state, coefficients,
        thresholds, and metadata.
    """
    prep = fit_preprocessor(X_cont, X_raw, seed=seed)
    X = transform_with_preprocessor(prep, X_cont, X_raw)

    parts = fit_l1_and_optional_refit(
        X,
        y,
        C=best["C"],
        seed=seed,
        refit_cache=None,
    )

    coef, intercept = combine_coefficients(parts, best["alpha"])

    if best["alpha"] > 0.0 and not parts["refit_used"]:
        warnings.warn(
            "Selected relaxed-refit candidate fell back to L1 on the full "
            "training set because its support was empty or exceeded "
            "MAX_REFIT_FEATURES. The same fallback rule was used in CV."
        )

    return {
        "preprocessor": prep,
        "coef": coef,
        "intercept": intercept,
        "score_threshold": best["score_threshold"],
        "probability_threshold": float(expit(best["score_threshold"])),
        "C": best["C"],
        "alpha": best["alpha"],
        "refit_l2_C": REFIT_L2_C,
        "refit_used": parts["refit_used"],
        "seed": seed,
        "cutoff_c": CUTOFF_C,
        "external_missing_raw_is_zero": EXTERNAL_MISSING_RAW_IS_ZERO,
        "label_definition": "healthy=1-MS; healthy is positive class",
    }


def predict_scores(model, X_cont, X_raw):
    """
    Compute raw linear decision scores.

    Parameters
    ----------
    model : dict
        Final model bundle.
    X_cont : pd.DataFrame
        Residual feature matrix.
    X_raw : pd.DataFrame
        Raw count feature matrix.

    Returns
    -------
    np.ndarray
        Raw linear scores.
    """
    X = transform_with_preprocessor(model["preprocessor"], X_cont, X_raw)
    return X @ model["coef"] + model["intercept"]

# ============================================================================
# Metrics and prediction export
# ============================================================================


def calculate_metrics(y_true, scores, score_threshold):
    """
    Compute evaluation metrics from decision scores.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    scores : array-like
        Decision scores.
    score_threshold : float
        Classification threshold on score scale.

    Returns
    -------
    dict
        Evaluation summary.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    y_pred = (scores >= score_threshold).astype(int)

    healthy = y_true == 1
    nonhealthy = y_true == 0
    has_two_classes = bool(healthy.any() and nonhealthy.any())

    return {
        "n_samples": int(len(y_true)),
        "n_healthy": int(healthy.sum()),
        "n_nonhealthy": int(nonhealthy.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred))
            if has_two_classes else np.nan
        ),
        "healthy_accuracy": (
            float(np.mean(y_pred[healthy] == 1))
            if healthy.any() else np.nan
        ),
        "nonhealthy_accuracy": (
            float(np.mean(y_pred[nonhealthy] == 0))
            if nonhealthy.any() else np.nan
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, scores))
            if has_two_classes else np.nan
        ),
        "score_threshold": float(score_threshold),
        "probability_threshold": float(expit(score_threshold)),
    }


def export_predictions(sample_ids, y_true, scores, score_threshold, out_csv, cutoff_c=CUTOFF_C):
    """
    Export per-sample prediction results and return summary metrics.

    Parameters
    ----------
    sample_ids : array-like
        Sample IDs.
    y_true : array-like
        True labels.
    scores : array-like
        Decision scores.
    score_threshold : float
        Classification threshold on score scale.
    out_csv : str
        Output CSV path.
    cutoff_c : float, default=CUTOFF_C
        Defer-zone half-width on centered decision score.

    Returns
    -------
    dict
        Evaluation summary.
    """
    scores = np.asarray(scores, dtype=float)
    decision_scores = scores - score_threshold
    y_pred = (scores >= score_threshold).astype(int)

    three_way = np.full(len(scores), "defer", dtype=object)
    three_way[decision_scores > cutoff_c] = "healthy"
    three_way[decision_scores < -cutoff_c] = "non-healthy"

    df = pd.DataFrame({
        "SampleID": sample_ids,
        "health_score": scores,
        "decision_score": decision_scores,
        "prob_healthy": expit(scores),
        "y_true_healthy": np.asarray(y_true, dtype=int),
        "y_pred_healthy": y_pred,
        "three_way_class": three_way,
        "score_threshold": score_threshold,
        "probability_threshold": float(expit(score_threshold)),
    })

    df.to_csv(out_csv, index=False)
    return calculate_metrics(y_true, scores, score_threshold)

# ============================================================================
# ROC and confusion matrix
# ============================================================================


def bootstrap_roc_ci(y, scores, seed):
    """
    Estimate pointwise ROC confidence bands and AUC confidence interval
    using stratified bootstrap resampling.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    scores : np.ndarray
        Decision scores.
    seed : int
        Random seed.

    Returns
    -------
    dict or None
        Bootstrap ROC summary, or None if only one class is present.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if pos_idx.size == 0 or neg_idx.size == 0:
        return None

    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 201)

    tpr_boot = np.empty((N_BOOTSTRAP, len(grid)))
    auc_boot = np.empty(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        idx = np.concatenate([
            rng.choice(pos_idx, size=pos_idx.size, replace=True),
            rng.choice(neg_idx, size=neg_idx.size, replace=True),
        ])

        yb = y[idx]
        sb = scores[idx]

        fpr, tpr, _ = roc_curve(yb, sb)

        unique_fpr, first_idx = np.unique(fpr, return_index=True)
        max_tpr = np.maximum.reduceat(tpr, first_idx)

        tpr_boot[b] = np.interp(
            grid,
            unique_fpr,
            max_tpr,
            left=0.0,
            right=1.0,
        )

        auc_boot[b] = roc_auc_score(yb, sb)

    return {
        "grid": grid,
        "low_tpr": np.percentile(tpr_boot, 2.5, axis=0),
        "high_tpr": np.percentile(tpr_boot, 97.5, axis=0),
        "auc_low": float(np.percentile(auc_boot, 2.5)),
        "auc_high": float(np.percentile(auc_boot, 97.5)),
    }


def plot_roc_and_confusion_matrix(y, scores, threshold, out_dir, suffix, seed):
    """
    Plot ROC curve and confusion matrix for a dataset.

    Parameters
    ----------
    y : np.ndarray
        True labels.
    scores : np.ndarray
        Decision scores.
    threshold : float
        Classification threshold.
    out_dir : str
        Output directory.
    suffix : str
        Suffix for output filenames.
    seed : int
        Random seed.

    Returns
    -------
    dict
        AUC confidence interval summary.
    """
    os.makedirs(out_dir, exist_ok=True)

    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    ci_result = {
        "auc_ci_low": np.nan,
        "auc_ci_high": np.nan,
    }

    if np.unique(y).size == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)

        ci = bootstrap_roc_ci(y, scores, seed)

        fig, ax = plt.subplots(figsize=(6.4, 5.6))

        ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", lw=1.2, label="Chance")

        if ci is not None:
            ax.fill_between(
                ci["grid"],
                ci["low_tpr"],
                ci["high_tpr"],
                color="#3b82f6",
                alpha=0.18,
                label="95% pointwise bootstrap band",
            )

            auc_label = (
                f"ROC AUC={auc:.3f}\n"
                f"95% CI {ci['auc_low']:.3f}-{ci['auc_high']:.3f}"
            )

            ci_result = {
                "auc_ci_low": ci["auc_low"],
                "auc_ci_high": ci["auc_high"],
            }
        else:
            auc_label = f"ROC AUC={auc:.3f}"

        ax.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=auc_label)

        tn, fp, fn, tp = cm.ravel()
        tpr_at_threshold = tp / (tp + fn)
        fpr_at_threshold = fp / (fp + tn)

        ax.scatter(
            fpr_at_threshold,
            tpr_at_threshold,
            s=65,
            color="#ef4444",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label=f"OOF-tuned cutoff\nTPR={tpr_at_threshold:.2f}, FPR={fpr_at_threshold:.2f}",
        )

        ax.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"roc_curve_{suffix}.png"), bbox_inches="tight")
        plt.close(fig)

    else:
        print(f"[{suffix}] Only one class is present; ROC AUC and balanced accuracy are reported as NaN.")

    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_totals,
        out=np.full(cm.shape, np.nan, dtype=float),
        where=row_totals > 0,
    )

    annotation = np.empty(cm.shape, dtype=object)
    for i in range(2):
        for j in range(2):
            percent = f"{100 * cm_norm[i, j]:.1f}%" if np.isfinite(cm_norm[i, j]) else "NA"
            annotation[i, j] = f"{cm[i, j]}\n({percent})"

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(
        cm,
        annot=annotation,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Non-healthy", "Healthy"],
        yticklabels=["Non-healthy", "Healthy"],
        ax=ax,
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_matrix_{suffix}.png"), bbox_inches="tight")
    plt.close(fig)

    return ci_result

# ============================================================================
# Coefficient export
# ============================================================================


def export_coefficients(model, stability, out_dir):
    """
    Export model coefficients and create a coefficient visualization.

    Parameters
    ----------
    model : dict
        Final model bundle.
    stability : pd.DataFrame
        Feature selection stability table from CV.
    out_dir : str
        Output directory.

    Returns
    -------
    pd.DataFrame
        Full coefficient table.
    """
    prep = model["preprocessor"]

    df = pd.DataFrame({
        "taxon": prep["feature_taxa"],
        "feature": prep["feature_names"],
        "feature_type": prep["feature_types"],
        "coefficient": model["coef"],
    })

    df["nonzero"] = np.abs(df["coefficient"]) > COEF_EPS
    df["sign"] = np.where(
        df["coefficient"] > COEF_EPS,
        "Positive",
        np.where(df["coefficient"] < -COEF_EPS, "Negative", "Zero"),
    )

    df = df.merge(stability, on="feature", how="left", validate="one_to_one")
    df = df.sort_values("coefficient", ascending=False).reset_index(drop=True)

    df.to_csv(os.path.join(out_dir, "coefficients.csv"), index=False)
    df.loc[df["nonzero"]].to_csv(os.path.join(out_dir, "coefficients_nonzero.csv"), index=False)

    plot_df = df.copy()
    plot_df["rank"] = np.arange(1, len(plot_df) + 1)

    colors = {
        "Positive": "#2ca02c",
        "Negative": "#d4a017",
        "Zero": "#bdbdbd",
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(0, color="#888888", linestyle="--", lw=1.0)

    for sign, sub in plot_df.groupby("sign", sort=False):
        ax.scatter(
            sub["rank"],
            sub["coefficient"],
            s=24 if sign != "Zero" else 12,
            color=colors[sign],
            edgecolors="white",
            linewidths=0.4,
            label=sign,
            alpha=0.9,
        )

    positive = plot_df.loc[plot_df["sign"] == "Positive"].head(N_ANNOTATE_POS)
    negative = plot_df.loc[plot_df["sign"] == "Negative"].tail(N_ANNOTATE_NEG)

    for sub, offset, va in [
        (positive, 9, "bottom"),
        (negative, -10, "top"),
    ]:
        for _, row in sub.iterrows():
            label = row["taxon"]
            if row["feature_type"] == "binary_presence":
                label += " (bin)"

            ax.annotate(
                label,
                (row["rank"], row["coefficient"]),
                xytext=(5, offset),
                textcoords="offset points",
                ha="left",
                va=va,
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
            )

    ax.set_xlabel("Features sorted by coefficient")
    ax.set_ylabel("Final linear coefficient")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coefficient_plot.png"), bbox_inches="tight")
    plt.close(fig)

    return df

# ============================================================================
# External validation
# ============================================================================


def prepare_external_inputs(model, X_cont, X_raw, out_dir, tag):
    """
    Align an external dataset to the internal training feature space.

    Missing residual columns are left as NaN so that they can be imputed
    using training medians during transformation. Missing raw columns are
    either filled with zero or treated as unknown, depending on the setting.

    Parameters
    ----------
    model : dict
        Final model bundle.
    X_cont : pd.DataFrame
        External residual matrix.
    X_raw : pd.DataFrame
        External raw count matrix.
    out_dir : str
        Output directory for the external dataset.
    tag : str
        External dataset tag.

    Returns
    -------
    tuple
        (aligned_X_cont, aligned_X_raw, n_missing_cont, n_missing_raw)
    """
    train_taxa = model["preprocessor"]["taxa_all"]

    missing_cont = [t for t in train_taxa if t not in X_cont.columns]
    missing_raw = [t for t in train_taxa if t not in X_raw.columns]

    report = pd.DataFrame({
        "taxon": train_taxa,
        "residual_column_missing": [t not in X_cont.columns for t in train_taxa],
        "raw_column_missing": [t not in X_raw.columns for t in train_taxa],
    })
    report.to_csv(os.path.join(out_dir, "feature_alignment_report.csv"), index=False)

    print(f"[{tag}] Missing training columns: residual={len(missing_cont)}, raw={len(missing_raw)}")

    Xc = X_cont.reindex(columns=train_taxa).copy()
    Xr = X_raw.reindex(columns=train_taxa).copy()

    if model["external_missing_raw_is_zero"] and missing_raw:
        Xr.loc[:, missing_raw] = 0.0
        print(f"[{tag}] Missing raw columns filled with 0 (interpreted as not detected).")
    elif missing_raw:
        print(f"[{tag}] Missing raw columns treated as unknown; presence imputed using training prevalence.")

    return Xc, Xr, len(missing_cont), len(missing_raw)


def run_external_validations(model, root_out_dir, internal_ids):
    """
    Evaluate the final model on all configured external datasets.

    Parameters
    ----------
    model : dict
        Final trained model bundle.
    root_out_dir : str
        Root output directory of the current model.
    internal_ids : array-like
        Internal sample IDs used to flag possible overlaps.

    Returns
    -------
    pd.DataFrame
        Summary of all external validation results.
    """
    results = []
    internal_id_set = set(map(str, internal_ids))

    for entry in EXTERNAL_DATASETS:
        tag = entry["tag"]
        folder = entry["path"]

        out_dir = os.path.join(root_out_dir, "external", tag)
        os.makedirs(out_dir, exist_ok=True)

        try:
            Xc, Xr, y, ids = load_dataset(
                os.path.join(folder, entry["residual"]),
                os.path.join(folder, entry["raw"]),
                os.path.join(folder, entry["metadata"]),
                external=True,
            )

            overlapping_ids = sorted(internal_id_set.intersection(map(str, ids)))
            if overlapping_ids:
                warnings.warn(
                    f"[{tag}] {len(overlapping_ids)} SampleID values also occur in the internal dataset. "
                    f"Verify whether these are duplicate biological samples or only ID collisions."
                )
                pd.DataFrame({"SampleID": overlapping_ids}).to_csv(
                    os.path.join(out_dir, "overlapping_sample_ids.csv"),
                    index=False,
                )

            Xc, Xr, n_missing_cont, n_missing_raw = prepare_external_inputs(
                model, Xc, Xr, out_dir, tag
            )

            scores = predict_scores(model, Xc, Xr)

            metrics = export_predictions(
                ids,
                y,
                scores,
                model["score_threshold"],
                os.path.join(out_dir, "predictions.csv"),
                cutoff_c=model["cutoff_c"],
            )

            ci_metrics = plot_roc_and_confusion_matrix(
                y,
                scores,
                model["score_threshold"],
                out_dir,
                suffix="external",
                seed=model["seed"],
            )

            metrics.update(ci_metrics)
            metrics.update({
                "dataset": tag,
                "status": "ok",
                "missing_residual_columns": n_missing_cont,
                "missing_raw_columns": n_missing_raw,
                "overlapping_internal_sample_ids": len(overlapping_ids),
            })

            pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

            with open(os.path.join(out_dir, "README_external.txt"), "w", encoding="utf-8") as f:
                f.write(f"External dataset: {tag}\n")
                f.write("Positive class: healthy = 1 - MS\n")
                f.write("Model and threshold are fixed from internal training.\n")
                f.write("No external labels are used to tune the model or threshold.\n")
                f.write("Bootstrap intervals condition on the fitted model and this evaluation cohort.\n\n")
                f.write(pd.Series(metrics).to_string())

            print(
                f"[External | {tag}] "
                f"N={len(y)}, "
                f"BA={metrics['balanced_accuracy']:.4f}, "
                f"AUC={metrics['roc_auc']:.4f}, "
                f"healthy acc={metrics['healthy_accuracy']:.4f}, "
                f"non-healthy acc={metrics['nonhealthy_accuracy']:.4f}"
            )

            results.append(metrics)

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[External | {tag}] FAILED: {message}")

            with open(os.path.join(out_dir, "ERROR.txt"), "w", encoding="utf-8") as f:
                f.write(message)

            results.append({
                "dataset": tag,
                "status": "failed",
                "error": message,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(root_out_dir, "external_summary.csv"), index=False)
    return result_df

# ============================================================================
# Single-seed execution
# ============================================================================


def run_one_seed(seed, X_cont, X_raw, y, sample_ids):
    """
    Run the full pipeline for one random seed.

    Parameters
    ----------
    seed : int
        Random seed.
    X_cont : pd.DataFrame
        Internal residual matrix.
    X_raw : pd.DataFrame
        Internal raw count matrix.
    y : np.ndarray
        Internal labels.
    sample_ids : np.ndarray
        Internal sample IDs.

    Returns
    -------
    tuple
        (summary_dict, external_summary_df)
    """
    start = time.perf_counter()

    seed_out_dir = os.path.join(OUT_DIR, f"seed_{seed}")
    model_out_dir = os.path.join(seed_out_dir, "logistic_regression")
    os.makedirs(model_out_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"Running optimized Logistic Regression | seed={seed}")
    print("=" * 72)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VALID_RATIO,
        random_state=seed,
    )

    tr, va = next(splitter.split(X_cont, y))

    Xc_train = X_cont.iloc[tr].copy()
    Xc_valid = X_cont.iloc[va].copy()
    Xr_train = X_raw.iloc[tr].copy()
    Xr_valid = X_raw.iloc[va].copy()

    y_train = y[tr]
    y_valid = y[va]

    id_train = sample_ids[tr]
    id_valid = sample_ids[va]

    split_assignment = np.full(len(y), "train", dtype=object)
    split_assignment[va] = "valid"

    pd.DataFrame({
        "SampleID": sample_ids,
        "split": split_assignment,
        "y_true_healthy": y,
    }).to_csv(os.path.join(seed_out_dir, "sample_split.csv"), index=False)

    print(
        f"Train: N={len(y_train)}, healthy={(y_train == 1).sum()}, non-healthy={(y_train == 0).sum()}"
    )
    print(
        f"Valid: N={len(y_valid)}, healthy={(y_valid == 1).sum()}, non-healthy={(y_valid == 0).sum()}"
    )

    best, oof_scores, cv_table, stability = select_model_by_cv(
        Xc_train,
        Xr_train,
        y_train,
        seed,
        model_out_dir,
    )

    print(f"\nSelected C={best['C']:.6g}, alpha={best['alpha']:.2f}")
    print(
        f"OOF score threshold={best['score_threshold']:.6f}, "
        f"probability threshold={expit(best['score_threshold']):.6f}"
    )

    oof_metrics = export_predictions(
        id_train,
        y_train,
        oof_scores,
        best["score_threshold"],
        os.path.join(model_out_dir, "oof_predictions.csv"),
    )

    model = fit_final_model(
        Xc_train,
        Xr_train,
        y_train,
        best,
        seed,
    )

    joblib.dump(
        model,
        os.path.join(model_out_dir, "model_bundle.joblib"),
        compress=3,
    )

    coef_df = export_coefficients(model, stability, model_out_dir)

    n_preprocessed = len(model["coef"])
    n_nonzero = int(np.count_nonzero(np.abs(model["coef"]) > COEF_EPS))
    n_taxa_used = int(coef_df.loc[coef_df["nonzero"], "taxon"].nunique())

    print(f"Preprocessed features: {n_preprocessed}")
    print(f"Final nonzero features: {n_nonzero}")
    print(f"Final unique taxa: {n_taxa_used}")
    print(f"L2 refit actually used: {model['refit_used']}")

    train_scores = predict_scores(model, Xc_train, Xr_train)
    valid_scores = predict_scores(model, Xc_valid, Xr_valid)

    train_metrics = export_predictions(
        id_train,
        y_train,
        train_scores,
        model["score_threshold"],
        os.path.join(model_out_dir, "train_predictions.csv"),
    )

    valid_metrics = export_predictions(
        id_valid,
        y_valid,
        valid_scores,
        model["score_threshold"],
        os.path.join(model_out_dir, "valid_predictions.csv"),
    )

    valid_ci = plot_roc_and_confusion_matrix(
        y_valid,
        valid_scores,
        model["score_threshold"],
        model_out_dir,
        suffix="valid",
        seed=seed,
    )
    valid_metrics.update(valid_ci)

    metric_table = pd.DataFrame([
        {"split": "train_apparent", **train_metrics},
        {"split": "train_oof_tuning", **oof_metrics},
        {"split": "valid", **valid_metrics},
    ])
    metric_table.to_csv(os.path.join(model_out_dir, "evaluation_metrics.csv"), index=False)

    selected_parameters = {
        "seed": int(seed),
        "C": float(model["C"]),
        "alpha": float(model["alpha"]),
        "refit_l2_C": float(model["refit_l2_C"]),
        "refit_used": bool(model["refit_used"]),
        "score_threshold": float(model["score_threshold"]),
        "probability_threshold": float(model["probability_threshold"]),
        "selection_metric": MODEL_SELECTION_METRIC,
        "actual_cv_splits": int(best["actual_cv_splits"]),
        "cv_repeats": int(best["cv_repeats"]),
        "preprocessed_feature_count": n_preprocessed,
        "final_model_feature_count": n_nonzero,
        "final_unique_taxa_count": n_taxa_used,
        "intercept": float(model["intercept"]),
        "external_missing_raw_is_zero": bool(EXTERNAL_MISSING_RAW_IS_ZERO),
    }

    with open(os.path.join(model_out_dir, "selected_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(selected_parameters, f, ensure_ascii=False, indent=2)

    print("\nHeld-out validation:")
    print(f"  Balanced accuracy: {valid_metrics['balanced_accuracy']:.4f}")
    print(f"  ROC AUC: {valid_metrics['roc_auc']:.4f}")
    print(f"  Healthy accuracy: {valid_metrics['healthy_accuracy']:.4f}")
    print(f"  Non-healthy accuracy: {valid_metrics['nonhealthy_accuracy']:.4f}")

    external_df = run_external_validations(
        model,
        model_out_dir,
        internal_ids=sample_ids,
    )
    external_df.insert(0, "seed", seed)

    elapsed = time.perf_counter() - start

    summary = {
        "seed": seed,
        "model": "logistic_regression",
        "C": model["C"],
        "alpha": model["alpha"],
        "refit_used": model["refit_used"],
        "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
        "valid_roc_auc": valid_metrics["roc_auc"],
        "valid_healthy_accuracy": valid_metrics["healthy_accuracy"],
        "valid_nonhealthy_accuracy": valid_metrics["nonhealthy_accuracy"],
        "valid_auc_ci_low": valid_metrics["auc_ci_low"],
        "valid_auc_ci_high": valid_metrics["auc_ci_high"],
        "threshold": model["probability_threshold"],
        "score_threshold": model["score_threshold"],
        "preprocessed_feature_count": n_preprocessed,
        "final_model_feature_count": n_nonzero,
        "final_unique_taxa_count": n_taxa_used,
        "elapsed_seconds": elapsed,
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(seed_out_dir, "model_summary.csv"),
        index=False,
    )

    with open(os.path.join(model_out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Optimized Logistic Regression pipeline\n")
        f.write("Positive class: healthy = 1 - MS\n")
        f.write("Random Forest is not used.\n\n")

        f.write("Selected parameters:\n")
        f.write(json.dumps(selected_parameters, ensure_ascii=False, indent=2))

        f.write("\n\nEvaluation:\n")
        f.write(metric_table.to_string(index=False))

        f.write("\n\nInterpretation notes:\n")
        f.write("- OOF predictions are used for model/threshold tuning; OOF tuning metrics are not independent test metrics.\n")
        f.write("- Validation and external labels are not used for tuning.\n")
        f.write("- health_score is the raw model decision score.\n")
        f.write("- decision_score = health_score - tuned_score_threshold.\n")
        f.write("- prob_healthy is expit(health_score); probability calibration is not established.\n")
        f.write("- three_way_class uses the centered decision_score.\n")
        f.write("- This custom score is not an official GMWI2 implementation.\n")
        f.write("- Bootstrap CIs condition on the fitted model and cohort; they do not include full retraining uncertainty.\n")
        f.write("- The supplied residual matrix must be checked for upstream preprocessing leakage.\n")
        f.write("- The default split assumes independent samples. Repeated subjects/families/sites may require grouped splits.\n")
        f.write(f"\nTotal elapsed seconds: {elapsed:.2f}\n")

    print(f"\nSeed {seed} total runtime: {elapsed:.1f}s")
    return summary, external_df

# ============================================================================
# Main
# ============================================================================


def main():
    """
    Main entry point of the pipeline.
    """
    if not RANDOM_SEEDS:
        raise ValueError("RANDOM_SEEDS must not be empty.")

    if not np.isclose(TRAIN_RATIO + VALID_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VALID_RATIO must equal 1.")

    print("Loading internal data...")

    X_cont, X_raw, y, sample_ids = load_dataset(
        BACTERIA_CSV,
        RAW_COUNT_CSV,
        META_CSV,
        external=False,
    )

    all_summaries = []
    all_external = []

    for seed in RANDOM_SEEDS:
        summary, external_df = run_one_seed(
            int(seed),
            X_cont,
            X_raw,
            y,
            sample_ids,
        )
        all_summaries.append(summary)
        all_external.append(external_df)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(OUT_DIR, "all_seeds_model_summary.csv"), index=False)

    metric_columns = [
        "valid_balanced_accuracy",
        "valid_roc_auc",
        "valid_healthy_accuracy",
        "valid_nonhealthy_accuracy",
        "threshold",
        "preprocessed_feature_count",
        "final_model_feature_count",
        "final_unique_taxa_count",
        "elapsed_seconds",
    ]

    mean_std = summary_df.groupby("model")[metric_columns].agg(["mean", "std"])
    mean_std.columns = [f"{metric}_{stat}" for metric, stat in mean_std.columns]
    mean_std = mean_std.reset_index()

    mean_std.to_csv(
        os.path.join(OUT_DIR, "all_seeds_model_summary_mean_std.csv"),
        index=False,
    )

    if all_external:
        external_all_df = pd.concat(all_external, ignore_index=True)
        external_all_df.to_csv(
            os.path.join(OUT_DIR, "all_seeds_external_summary.csv"),
            index=False,
        )

    print("\n" + "=" * 72)
    print("Overall validation summary")
    print("=" * 72)

    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nMean / std across seeds:")
    print(mean_std.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if len(RANDOM_SEEDS) == 1:
        print("\nNote: only one seed was used; across-seed standard deviations are NaN.")

    with open(os.path.join(OUT_DIR, "README_all_seeds_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Optimized Logistic Regression only\n")
        f.write(f"Seeds: {RANDOM_SEEDS}\n")
        f.write(f"Residual input: {BACTERIA_CSV}\n")
        f.write(f"Raw count input: {RAW_COUNT_CSV}\n")
        f.write(f"Metadata input: {META_CSV}\n")
        f.write(f"Model selection metric: {MODEL_SELECTION_METRIC}\n")
        f.write("\nDo not select a preferred random seed using validation or external performance and then report only that seed.\n")

        f.write("\nPer-seed validation results:\n")
        f.write(summary_df.to_string(index=False))

        f.write("\n\nAcross-seed descriptive mean/std:\n")
        f.write(mean_std.to_string(index=False))

    print(f"\nOutputs saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()