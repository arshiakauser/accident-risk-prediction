# -*- coding: utf-8 -*-

import argparse
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)

from preprocessing8 import (POST_EVENT_COLUMNS, TARGET, build_preprocessing_pipeline_shap,
                            recommend_action, risk_level_from_proba, transform_test)

# Optional boosting libraries: a missing one is skipped with a message instead of crashing
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_PATH = "accidents_dataset.csv"
MODEL_PATH = "best_model.pkl"
PREPROCESSING_PATH = "preprocessing_artifacts.pkl"
RESULTS_XLSX = "model_results_ranked.xlsx"
RESULTS_CSV = "model_results.csv"
CHART_PATH = "model_performance_comparison.png"
RANDOM_STATE = 42

# How the alert threshold is chosen (always from out-of-fold predictions on the TRAINING data):
#   "f1"           : threshold with the best F1 (balances missed accidents and false alarms)
#   "precision"    : lowest threshold whose precision is >= TARGET_PRECISION  (fewer false alarms)
#   "alert_budget" : flag only the top ALERT_BUDGET share of rows by risk      (fixed workload)
# Business constraint = "minimize false alarms"  ->  use "alert_budget" or "precision".
THRESHOLD_MODE = "alert_budget"
TARGET_PRECISION = 0.15
ALERT_BUDGET = 0.05

# How the BEST MODEL is chosen (from out-of-fold training results only, never the test set):
#   "CV_P@5%"  : precision among the 5% riskiest rows  -> use with ALERT_BUDGET = 0.05
#   "CV_P@10%" : same at 10% (steadier with few accidents) -> use with ALERT_BUDGET = 0.10
#   "CV_F1", "CV_PR_AUC", "CV_Capture@5%", "CV_Capture@10%" are also available.
# Keep the % in the metric equal to your alert budget.
SELECTION_METRIC = "CV_P@5%"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def resampler_factory(method="smote", sampling_strategy=0.3, random_state=RANDOM_STATE):
    """Create a resampler. sampling_strategy = minority/majority ratio AFTER resampling."""
    method = method.strip().lower()
    makers = {
        "smote": SMOTE, "adasyn": ADASYN, "smote-enn": SMOTEENN,
        "borderlinesmote": BorderlineSMOTE, "smotetomek": SMOTETomek,
    }
    if method not in makers:
        raise ValueError(f"Unknown resampling method: {method}. Choose from {list(makers)}")
    return makers[method](sampling_strategy=sampling_strategy, random_state=random_state)


def f1_threshold(y_true, y_probs):
    """Threshold that maximizes F1 (computed from the precision-recall curve)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # last PR point has no threshold
    return float(thresholds[best]), float(f1[best])


def compute_metrics_dict(name, threshold, y_true, y_pred, y_probs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Threshold": threshold,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_probs),
        "PR_AUC": average_precision_score(y_true, y_probs),
        "Accuracy": accuracy_score(y_true, y_pred),  # kept for completeness, do not rely on it
        "True_Negative": tn, "False_Positive": fp, "False_Negative": fn, "True_Positive": tp,
    }


def bootstrap_ci(y_true, y_probs, threshold, n_boot=500, seed=RANDOM_STATE):
    """95% bootstrap intervals for test F1 and PR-AUC (essential when the test set has few positives)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(y_probs)
    f1s, aps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        f1s.append(f1_score(y[idx], (p[idx] >= threshold).astype(int), zero_division=0))
        aps.append(average_precision_score(y[idx], p[idx]))
    fmt = lambda v: f"{np.percentile(v, 2.5):.2f}-{np.percentile(v, 97.5):.2f}"
    return {"F1_95CI": fmt(f1s), "PR_AUC_95CI": fmt(aps)}


def topk_report(y_true, probs, fractions=(0.05, 0.10, 0.20), prefix=""):
    """If only the top X% highest-risk rows are flagged: how many flags are real accidents
    (P@X = precision) and what share of all accidents is caught (Capture@X = recall)?
    Lift@10% = P@10% divided by the overall accident rate (1.0 = no better than random)."""
    y, p = np.asarray(y_true), np.asarray(probs)
    order = np.argsort(-p, kind="stable")
    base, total_pos = y.mean(), y.sum()
    out = {}
    for frac in fractions:
        k = max(1, int(round(len(y) * frac)))
        hits = y[order[:k]].sum()
        tag = f"{int(round(frac * 100))}%"
        out[f"{prefix}P@{tag}"] = hits / k
        out[f"{prefix}Capture@{tag}"] = hits / total_pos if total_pos else np.nan
    out[f"{prefix}Lift@10%"] = out[f"{prefix}P@10%"] / base if base > 0 else np.nan
    return out


def select_threshold(y_true, oof_probs, mode="f1", target_precision=0.15, alert_budget=0.05):
    """Choose the alert threshold from out-of-fold predictions. Returns (threshold, description)."""
    if mode not in ("f1", "precision", "alert_budget"):
        raise ValueError("threshold_mode must be 'f1', 'precision' or 'alert_budget'")
    oof = np.asarray(oof_probs)

    if mode == "f1":
        thr, _ = f1_threshold(y_true, oof)
        return thr, "best F1 on out-of-fold predictions"

    if mode == "precision":
        precision, recall, thresholds = precision_recall_curve(y_true, oof)
        flagged = len(oof) - np.searchsorted(np.sort(oof), thresholds, side="left")
        min_flagged = max(5, int(0.01 * len(oof)))   # ignore thresholds that flag only a handful of rows
        ok = np.where((precision[:-1] >= target_precision) & (flagged >= min_flagged))[0]
        if ok.size:
            best = ok[np.argmax(recall[:-1][ok])]    # most accidents caught while keeping the precision target
            return float(thresholds[best]), f"lowest threshold with out-of-fold precision >= {target_precision:.0%}"
        warnings.warn(f"No threshold reaches {target_precision:.0%} precision on the out-of-fold predictions "
                      f"- falling back to alert_budget={alert_budget:.0%}.")

    thr = float(np.quantile(oof, 1 - alert_budget))
    return thr, f"flags the top {alert_budget:.0%} highest-risk rows (out-of-fold)"


def _grid_size(params):
    size = 1
    for values in params.values():
        size *= len(values)
    return size


def get_model_configs(random_state=RANDOM_STATE):
    """Model + search space + the ONE balancing strategy used for it."""
    rs = random_state
    cfg = {
        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=rs),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
            "balancing": "class_weight",
        },
        "BalancedRandomForest": {
            "model": BalancedRandomForestClassifier(sampling_strategy="all", replacement=True,
                                                    bootstrap=False, n_jobs=1, random_state=rs),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10],
            },
            "balancing": "internal",
        },
        "GradientBoost": {
            "model": GradientBoostingClassifier(random_state=rs),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__subsample": [0.7, 0.8, 1.0],
            },
            "balancing": "sampler",
        },
    }
    boosting_params = {
        "classifier__n_estimators": [200, 500],
        "classifier__max_depth": [3, 5, 7, 9],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    }
    if LGBMClassifier is not None:
        cfg["LightGBM"] = {"model": LGBMClassifier(n_jobs=1, random_state=rs, verbose=-1),
                           "params": dict(boosting_params), "balancing": "sampler"}
    if CatBoostClassifier is not None:
        cfg["CatBoost"] = {
            "model": CatBoostClassifier(verbose=0, thread_count=1, random_state=rs,
                                        allow_writing_files=False),
            "params": {
                "classifier__iterations": [200, 500],
                "classifier__depth": [4, 6, 8],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__l2_leaf_reg": [1, 3, 5, 7],
            },
            "balancing": "sampler",
        }
    return cfg


def make_pipeline(model_cfg, resampling, sampling_strategy, random_state=RANDOM_STATE):
    """imblearn Pipeline: the sampler (if any) only runs on training folds."""
    steps = []
    if model_cfg["balancing"] == "sampler":
        steps.append(("sampler", resampler_factory(resampling, sampling_strategy, random_state)))
    steps.append(("classifier", clone(model_cfg["model"])))
    return ImbPipeline(steps)


# ----------------------------------------------------------------------------
# Data loading + split
# ----------------------------------------------------------------------------
def load_and_split(data_path=DATA_PATH, test_size=0.2, time_col="timestamp", split_mode="auto",
                   random_state=RANDOM_STATE):
    """Split into train / test.

    split_mode: "auto"       -> time-based if a usable timestamp column exists, else stratified
                "time"       -> train on the past, test on the latest part (needs a timestamp)
                "stratified" -> random stratified split
    """
    if split_mode not in ("auto", "time", "stratified"):
        raise ValueError("split_mode must be 'auto', 'time' or 'stratified'")
    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {data_path}")
    df[TARGET] = df[TARGET].astype(int)   # the CSV may store True/False

    train_df = test_df = None
    split_desc = "stratified random"
    if split_mode in ("auto", "time"):
        ts = pd.to_datetime(df[time_col], errors="coerce") if time_col in df.columns else None
        if ts is not None and ts.notna().mean() > 0.95:
            ordered = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts", kind="stable").drop(columns="_ts")
            cut = int(len(ordered) * (1 - test_size))
            cand_train, cand_test = ordered.iloc[:cut], ordered.iloc[cut:]
            if cand_train[TARGET].nunique() == 2 and cand_test[TARGET].nunique() == 2:
                train_df, test_df = cand_train, cand_test
                split_desc = f"time-based (test = latest {test_size:.0%} of '{time_col}')"
            else:
                warnings.warn("Time-based split left one class missing - using a stratified split instead.")
        elif split_mode == "time":
            raise ValueError(f"split_mode='time' needs a parseable '{time_col}' column")

    if train_df is None:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[TARGET],
                                             random_state=random_state)

    def to_xy(frame):
        return frame.drop(columns=POST_EVENT_COLUMNS, errors="ignore"), frame[TARGET].astype(int)

    X_train, y_train = to_xy(train_df)
    X_test, y_test = to_xy(test_df)

    print(f"📊 Dataset: {len(df):,} rows | positives: {int(df[TARGET].sum())} ({df[TARGET].mean() * 100:.2f}%)")
    print(f"✂️  Split: {split_desc}")
    print(f"✅ Train: {X_train.shape} ({int(y_train.sum())} positives) | "
          f"Test: {X_test.shape} ({int(y_test.sum())} positives)")
    if y_test.sum() < 30:
        print(f"⚠️  The test set has only {int(y_test.sum())} positive rows. Test metrics are very noisy - "
              "read the 95% intervals (F1_95CI, PR_AUC_95CI) and rely on CV scores for model comparison.")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_models(X_train, y_train, X_test, y_test, resampling="smote", sampling_strategy=0.3,
                 cv_splits=5, n_iter=20, models=None, selection_metric=SELECTION_METRIC,
                 model_path=MODEL_PATH, random_state=RANDOM_STATE,
                 threshold_mode=THRESHOLD_MODE, target_precision=TARGET_PRECISION,
                 alert_budget=ALERT_BUDGET):
    """Tune every model with CV, pick threshold + best model WITHOUT the test set,
    then evaluate on the test set once.

    selection_metric : how the best model is chosen (from out-of-fold training results only):
                       "CV_P@5%", "CV_P@10%", "CV_Capture@5%", "CV_Capture@10%", "CV_F1" or "CV_PR_AUC".
    models           : optional list of model names to run (e.g. for a quick test).
    threshold_mode   : "f1", "precision" or "alert_budget" (see the config block at the top).
    """
    print("\n" + "=" * 60)
    print("📊 CLASS DISTRIBUTION (training set)")
    print("=" * 60)
    for cls, cnt in zip(*np.unique(y_train, return_counts=True)):
        print(f"  Class {cls}: {cnt:,} samples ({cnt / len(y_train) * 100:.2f}%)")

    configs = get_model_configs(random_state)
    if models is not None:
        unknown = [m for m in models if m not in configs]
        if unknown:
            print(f"⚠️  Unknown / unavailable models ignored: {unknown}")
        configs = {k: v for k, v in configs.items() if k in models}

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results, fitted = [], {}

    for name, cfg in configs.items():
        print(f"\n{'=' * 60}\n🚀 Training {name}  (balancing: {cfg['balancing']})\n{'=' * 60}")
        pipe = make_pipeline(cfg, resampling, sampling_strategy, random_state)
        search = RandomizedSearchCV(
            pipe, cfg["params"], n_iter=min(n_iter, _grid_size(cfg["params"])), cv=skf,
            scoring="average_precision", n_jobs=-1, random_state=random_state,
            error_score="raise", verbose=1)
        try:
            search.fit(X_train, y_train)  # ORIGINAL training data; resampling happens inside each fold
        except Exception as exc:
            print(f"⚠️  Skipping {name} due to error: {exc}")
            continue

        best_pipe = search.best_estimator_
        print(f"✅ Best params: {search.best_params_}")
        print(f"✅ CV PR-AUC: {search.best_score_:.4f}")

        # Threshold from OUT-OF-FOLD probabilities on the training data (test set not used)
        oof = cross_val_predict(clone(best_pipe), X_train, y_train, cv=skf,
                                method="predict_proba", n_jobs=-1)[:, 1]
        _, cv_f1 = f1_threshold(y_train, oof)   # best achievable out-of-fold F1, reported for every model
        threshold, rule = select_threshold(y_train, oof, threshold_mode, target_precision, alert_budget)
        cv_flag = oof >= threshold
        print(f"✅ Threshold {threshold:.4f} ({rule}); best out-of-fold F1 = {cv_f1:.4f}")

        # Test set: used once, with the already-fixed threshold
        test_probs = best_pipe.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= threshold).astype(int)
        row = compute_metrics_dict(name, threshold, y_test, test_pred, test_probs)
        row.update({"Balancing": cfg["balancing"], "CV_PR_AUC": search.best_score_, "CV_F1": cv_f1,
                    "CV_Precision_at_threshold": precision_score(y_train, cv_flag, zero_division=0),
                    "CV_Recall_at_threshold": recall_score(y_train, cv_flag, zero_division=0),
                    "Alert_Rate": float(test_pred.mean()),
                    **topk_report(y_train, oof, prefix="CV_"), **topk_report(y_test, test_probs),
                    **bootstrap_ci(y_test, test_probs, threshold), "Best_Params": str(search.best_params_)})
        results.append(row)
        fitted[name] = (best_pipe, threshold, cfg["balancing"], rule)
        print(f"📊 Test: F1={row['F1']:.3f} (95% CI {row['F1_95CI']})  Recall={row['Recall']:.3f}  "
              f"Precision={row['Precision']:.3f}  PR-AUC={row['PR_AUC']:.3f} (95% CI {row['PR_AUC_95CI']})")
        print(f"   If only the top 10% riskiest rows are flagged: precision {row['P@10%']:.3f}, "
              f"catches {row['Capture@10%']:.0%} of accidents (lift {row['Lift@10%']:.1f}x)")

    if not results:
        raise RuntimeError("No model trained successfully.")

    results_df = pd.DataFrame(results).sort_values(by=selection_metric, ascending=False).reset_index(drop=True)
    best_name = results_df.loc[0, "Model"]
    best_pipe, best_threshold, balancing, best_rule = fitted[best_name]
    best_row = results_df.loc[0]
    topk_keys = ("P@5%", "Capture@5%", "P@10%", "Capture@10%", "Lift@10%")

    joblib.dump({
        "pipeline": best_pipe,
        "model_name": best_name,
        "threshold": best_threshold,
        "threshold_mode": threshold_mode,
        "threshold_rule": best_rule,
        "balancing": balancing,
        "resampling_method": resampling if balancing == "sampler" else None,
        "sampling_strategy": sampling_strategy if balancing == "sampler" else None,
        "feature_names": list(X_train.columns),
        "cv_metrics": {"CV_F1": float(best_row["CV_F1"]), "CV_PR_AUC": float(best_row["CV_PR_AUC"])},
        "test_metrics": {k: float(best_row[k]) for k in ("F1", "Recall", "Precision", "ROC_AUC", "PR_AUC") + topk_keys},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }, model_path)

    print(f"\n{'=' * 60}")
    print(f"✅ BEST MODEL (by {selection_metric}): {best_name}")
    print(f"   CV F1 {best_row['CV_F1']:.4f} | Test F1 {best_row['F1']:.4f} | threshold {best_threshold:.4f}")
    print(f"   Threshold rule: {best_rule}")
    print(f"   Saved as: {model_path}")
    print("=" * 60)
    return results_df


def plot_results(results_df, path=CHART_PATH, show=True):
    """Bar chart of the metrics that matter for a rare event (no accuracy)."""
    plot_df = results_df.sort_values("PR_AUC", ascending=False)
    metrics = ["PR_AUC", "F1", "Recall", "Precision"]
    width = 0.8 / len(metrics)
    x = np.arange(len(plot_df))
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - (len(metrics) - 1) / 2) * width, plot_df[metric], width=width, label=metric)
    plt.xticks(x, plot_df["Model"], rotation=45, ha="right")
    plt.ylabel("Score (test set)")
    plt.title("Model comparison on the held-out test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"📊 Chart saved as '{path}'")
    if show:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------
# End-to-end training run
# ----------------------------------------------------------------------------
def run_training(data_path=DATA_PATH, n_features_select=25, use_sfs=False, resampling="smote",
                 sampling_strategy=0.3, cv_splits=5, n_iter=20, models=None, show_plot=True,
                 split_mode="auto", model_path=MODEL_PATH, preprocessing_path=PREPROCESSING_PATH,
                 threshold_mode=THRESHOLD_MODE, target_precision=TARGET_PRECISION,
                 alert_budget=ALERT_BUDGET, selection_metric=SELECTION_METRIC):
    X_train, X_test, y_train, y_test = load_and_split(data_path, split_mode=split_mode)

    # Less than 4 weeks of data cannot teach a weekly pattern, and the test days fall on weekdays the
    # model has never seen -> day_of_week would only act as a clock. Keep it out of the model.
    extra_drop = []
    if "timestamp" in X_train.columns and "day_of_week" in X_train.columns:
        ts = pd.to_datetime(X_train["timestamp"], errors="coerce").dropna()
        if len(ts) and (ts.max() - ts.min()).days < 28:
            extra_drop.append("day_of_week")
            print("ℹ️  Training data covers less than 4 weeks -> 'day_of_week' is not used as a feature.")

    print("\n🔧 Building preprocessing pipeline (fitted on the training split only)...")
    X_train_sel, selected_features, feature_engineer, preprocessor = build_preprocessing_pipeline_shap(
        X_train, y_train, n_features_select=n_features_select, use_sfs=use_sfs,
        extra_drop_columns=extra_drop)
    X_test_sel = transform_test(X_test, feature_engineer, preprocessor, selected_features,
                                list(X_train_sel.columns))

    joblib.dump({
        "feature_engineer": feature_engineer,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "train_columns": list(X_train_sel.columns),
    }, preprocessing_path)

    print("\n🚀 STARTING MODEL TRAINING")
    results_df = train_models(X_train_sel, y_train, X_test_sel, y_test, resampling=resampling,
                              sampling_strategy=sampling_strategy, cv_splits=cv_splits,
                              n_iter=n_iter, models=models, model_path=model_path,
                              threshold_mode=threshold_mode, target_precision=target_precision,
                              alert_budget=alert_budget, selection_metric=selection_metric)

    show_cols = ["Model", "Balancing", "CV_F1", "CV_PR_AUC", "F1", "F1_95CI", "Recall", "Precision",
                 "PR_AUC", "PR_AUC_95CI", "Threshold"]
    print(f"\n📊 MODELS RANKED BY {selection_metric} (test metrics are for reporting only)")
    print(results_df[show_cols].round(4).to_string(index=False))

    topk_cols = ["Model", "Alert_Rate", "CV_P@10%", "CV_Capture@10%", "P@5%", "Capture@5%",
                 "P@10%", "Capture@10%", "Lift@10%"]
    print("\n📊 IF ONLY THE RISKIEST ROWS ARE FLAGGED (P@x = share of flags that are real accidents, "
          "Capture@x = share of all accidents caught; accident rate in test = "
          f"{y_test.mean():.1%})")
    print(results_df[topk_cols].round(3).to_string(index=False))

    results_df.to_excel(RESULTS_XLSX, index=False)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ Results saved to '{RESULTS_XLSX}' and '{RESULTS_CSV}'")
    plot_results(results_df, show=show_plot)
    return results_df


# ----------------------------------------------------------------------------
# Prediction on new data
# ----------------------------------------------------------------------------
def predict_new_data(new_data_path, model_path=MODEL_PATH, preprocessing_path=PREPROCESSING_PATH,
                     output_xlsx="prediction_results.xlsx", output_csv="prediction_results.csv"):
    """Score a CSV. The target / post-accident columns are optional in the file."""
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    threshold = artifact.get("threshold", 0.5)
    print(f"✅ Loaded model: {artifact.get('model_name', model_path)} | threshold {threshold:.4f} "
          f"({artifact.get('threshold_rule', 'rule not stored')})")

    pre = joblib.load(preprocessing_path)
    new_data = pd.read_csv(new_data_path)
    clean = new_data.drop(columns=POST_EVENT_COLUMNS, errors="ignore")

    X_new = transform_test(clean, pre["feature_engineer"], pre["preprocessor"],
                           pre["selected_features"], pre["train_columns"])
    proba = pipeline.predict_proba(X_new)[:, 1]
    preds = (proba >= threshold).astype(int)
    levels = risk_level_from_proba(proba, threshold)

    results = clean.copy()
    results["Accident_Probability"] = np.round(proba, 4)
    results["Prediction"] = preds
    results["Risk_Level"] = levels
    results["Recommended_Action"] = [recommend_action(l) for l in levels]

    print("\n📊 Risk distribution (%):")
    print(results["Risk_Level"].value_counts(normalize=True).mul(100).round(2).to_string())

    if TARGET in new_data.columns:  # labels available -> report real performance
        y_true = new_data[TARGET].astype(int)
        print(f"\n📈 Labels found: F1={f1_score(y_true, preds, zero_division=0):.3f}  "
              f"Recall={recall_score(y_true, preds, zero_division=0):.3f}  "
              f"Precision={precision_score(y_true, preds, zero_division=0):.3f}")
        print("   ⚠️ Only meaningful for rows the model has NEVER seen. If this file contains training rows "
              "the scores are inflated - quote the test / CV scores from training instead.")

    results.to_excel(output_xlsx, index=False)
    results.to_csv(output_csv, index=False)
    print(f"\n💾 Predictions saved as '{output_xlsx}' and '{output_csv}'")
    return results


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or use the accident-risk model")
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "predict"])
    parser.add_argument("--data", default=DATA_PATH, help="training CSV (train mode)")
    parser.add_argument("--new-data", default="new_data.csv", help="CSV to score (predict mode)")
    parser.add_argument("--split", default="auto", choices=["auto", "time", "stratified"],
                        help="how to split train/test (train mode)")
    parser.add_argument("--threshold-mode", default=THRESHOLD_MODE, choices=["f1", "precision", "alert_budget"],
                        help="how the alert threshold is chosen (train mode)")
    parser.add_argument("--target-precision", type=float, default=TARGET_PRECISION,
                        help="used with --threshold-mode precision")
    parser.add_argument("--alert-budget", type=float, default=ALERT_BUDGET,
                        help="used with --threshold-mode alert_budget (share of rows flagged)")
    parser.add_argument("--selection-metric", default=SELECTION_METRIC,
                        choices=["CV_F1", "CV_PR_AUC", "CV_P@5%", "CV_P@10%", "CV_Capture@5%", "CV_Capture@10%"],
                        help="how the best model is chosen (train mode)")
    args, _ = parser.parse_known_args()  # tolerant of IDE-injected arguments

    if args.mode == "predict":
        results = predict_new_data(args.new_data)
        print(results.head())
    else:
        run_training(data_path=args.data, split_mode=args.split, threshold_mode=args.threshold_mode,
                     target_precision=args.target_precision, alert_budget=args.alert_budget,
                     selection_metric=args.selection_metric)
        print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
