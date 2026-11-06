# -*- coding: utf-8 -*-

import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# ------------------------------- config -------------------------------
DATA_PATH = "accidents_dataset.csv"
TARGET = "accident_occurred"
OUTPUT_DIR = "eda_outputs"
SHOW_PLOTS = False            # True -> also display figures (blocks until you close them)
MAX_LEVELS_PLOTTED = 15       # high-cardinality columns: only the most frequent levels are plotted
AUTO_ADD_CATEGORICALS = True  # also analyse other low-cardinality text columns

# Only known AFTER an accident -> must never be used as predictors
LEAKAGE_COLUMNS = ["severity", "cause", "vehicles_involved", "veh_count_at_accident"]

CATEGORICAL_COLS = [
    "state", "road_type", "has_signal", "enforcement_level", "season",
    "day_of_week", "lighting", "weather", "is_peak", "peak",
    "traffic_data_quality_flag", "signal_status", "signal_data_quality_flag",
]
ID_LIKE_COLS = ["timestamp", "intersection_id", "location_id"]  # never analysed as categoricals

sns.set_theme(style="whitegrid")


# ------------------------------- helpers -------------------------------
def save_fig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{safe}.png"), dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def benjamini_hochberg(p_values):
    """Adjusted p-values (False Discovery Rate). NaN stays NaN."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return adjusted
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(ranked, 0, 1)
    adjusted[valid] = out
    return adjusted


def cramers_v(table):
    """Bias-corrected Cramer's V (0 = no association, 1 = perfect association)."""
    r, k = table.shape
    if r < 2 or k < 2:
        return np.nan
    n = table.values.sum()
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan


# ------------------------------- overview -------------------------------
def basic_overview(df):
    print("### Basic Info ###")
    df.info()
    print("\n### Summary Statistics ###")
    print(df.describe(include="all").T.to_string())
    print("\n### Missing Values ###")
    print(df.isnull().sum().sort_values(ascending=False).to_string())


def plot_target_balance(df):
    counts = df[TARGET].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    print(f"\n### Target balance ###\n{counts.to_string()}\nPositive rate: {pct.get(1, 0):.2f}%")
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4C78A8")
    for i, (c, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, c, f"{c:,}\n({p:.1f}%)", ha="center", va="bottom")
    plt.title(f"Target balance: {TARGET}")
    plt.xlabel(TARGET)
    plt.ylabel("Rows")
    save_fig("00_target_balance")


# ------------------------------- numeric -------------------------------
def numeric_univariate(df, num_cols):
    rows = []
    ts = None
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().mean() > 0.9:
            ts = parsed
    freq = "D"
    if ts is not None and (ts.max() - ts.min()).days > 180:
        freq = "W"

    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        mode = series.mode()
        rows.append({
            "Feature": col,
            "Missing_%": round(df[col].isna().mean() * 100, 2),
            "Mean": series.mean(), "Median": series.median(),
            "Mode": mode.iloc[0] if not mode.empty else np.nan,
            "Variance": series.var(), "StdDev": series.std(),
            "Range": series.max() - series.min(),
            "Skewness": series.skew(), "Kurtosis": series.kurt(),
        })

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        sns.histplot(series, kde=True, bins=30, ax=axes[0])
        axes[0].set_title(f"Histogram - {col}")
        sns.boxplot(x=series, ax=axes[1])
        axes[1].set_title(f"Boxplot - {col}")
        save_fig(f"num_uni_{col}")

        if ts is not None:
            valid = ts.notna()
            trend = df.loc[valid, col].groupby(ts[valid]).mean().resample(freq).mean()
            plt.figure(figsize=(8, 3.5))
            trend.plot()
            plt.title(f"Time plot ({'daily' if freq == 'D' else 'weekly'} mean) - {col}")
            save_fig(f"num_time_{col}")

    summary = pd.DataFrame(rows).round(3)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_excel(os.path.join(OUTPUT_DIR, "EDA_Univariate_Stats.xlsx"), index=False)
    print("\n### Numeric univariate summary ###")
    print(summary.to_string(index=False))
    return summary


def numeric_vs_target(df, num_cols):
    y = df[TARGET]
    if y.nunique() != 2:
        print("Target is not binary - numeric-vs-target tests skipped.")
        return pd.DataFrame()

    rows = []
    for col in num_cols:
        tmp = df[[col, TARGET]].dropna()
        x1 = tmp.loc[tmp[TARGET] == 1, col]
        x0 = tmp.loc[tmp[TARGET] == 0, col]
        row = {"Feature": col, "Median_no_accident": x0.median(), "Median_accident": x1.median(),
               "AUC": np.nan, "Cohens_d": np.nan, "MWU_p": np.nan}
        if len(x1) > 1 and len(x0) > 1 and tmp[col].nunique() > 1:
            u, p = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            row["AUC"] = u / (len(x1) * len(x0))  # P(accident row has the larger value); 0.5 = no signal
            row["MWU_p"] = p
            pooled = np.sqrt(((len(x1) - 1) * x1.var() + (len(x0) - 1) * x0.var()) / (len(x1) + len(x0) - 2))
            row["Cohens_d"] = (x1.mean() - x0.mean()) / pooled if pooled > 0 else np.nan
        rows.append(row)

    res = pd.DataFrame(rows)
    res["MWU_p_adj_BH"] = benjamini_hochberg(res["MWU_p"].to_numpy())
    res["Signal_strength"] = (res["AUC"] - 0.5).abs() * 2   # 0 = none, 1 = perfect separation
    res = res.sort_values("Signal_strength", ascending=False).round(4)
    res.to_excel(os.path.join(OUTPUT_DIR, "NumVsTarget_AccidentOccurred.xlsx"), index=False)
    print("\n### Numeric features vs target (sorted by effect size, read AUC / Cohen's d, not p) ###")
    print(res.to_string(index=False))

    # correlation heat map + highly correlated pairs
    corr = df[num_cols].corr()
    size = max(6, min(0.45 * len(num_cols), 16))
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(corr, annot=len(num_cols) <= 15, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation heatmap of numerical columns")
    save_fig("num_correlation_heatmap")

    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
             .rename("corr").reset_index().rename(columns={"level_0": "Feature_1", "level_1": "Feature_2"}))
    strong = pairs[pairs["corr"].abs() > 0.9].sort_values("corr", key=abs, ascending=False)
    if not strong.empty:
        print("\n⚠️ Highly correlated numeric pairs (|r| > 0.9) - candidates to drop one of each:")
        print(strong.to_string(index=False))
    return res


# ------------------------------- categorical -------------------------------
def get_categorical_columns(df):
    cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    missing = [c for c in CATEGORICAL_COLS if c not in df.columns]
    if missing:
        print(f"ℹ️ Categorical columns not in the data (skipped): {missing}")
    if AUTO_ADD_CATEGORICALS:
        skip = set(cols) | set(LEAKAGE_COLUMNS) | set(ID_LIKE_COLS) | {TARGET}
        extra = [c for c in df.columns
                 if c not in skip and df[c].nunique() <= 30 and not pd.api.types.is_float_dtype(df[c])
                 and (pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
                      or pd.api.types.is_string_dtype(df[c])
                      or isinstance(df[c].dtype, pd.CategoricalDtype))]
        if extra:
            print(f"ℹ️ Also analysing other low-cardinality columns: {extra}")
        cols += extra
    return cols


def univariate_categorical(df, col):
    counts = df[col].value_counts(dropna=False)
    print(f"\n===== Univariate: {col} =====")
    print(pd.DataFrame({"count": counts, "pct": (counts / counts.sum() * 100).round(2)}).to_string())
    top = counts.head(MAX_LEVELS_PLOTTED)
    plt.figure(figsize=(8, 4.5))
    top.rename(index=str).plot(kind="bar", color="#4C78A8")
    plt.title(f"Frequency of {col}" + (f" (top {MAX_LEVELS_PLOTTED})" if len(counts) > MAX_LEVELS_PLOTTED else ""))
    plt.xticks(rotation=45, ha="right")
    save_fig(f"cat_uni_{col}")


def bivariate_categorical(df, col):
    """Chi-square + Cramer's V + accident rate per level. Returns one summary row."""
    # NaN is a real category in this data (e.g. "no signal", "no glitch"), so keep it as "missing"
    values = df[col].astype(object).where(df[col].notna(), "missing")
    table = pd.crosstab(values, df[TARGET])
    row = {"Feature": col, "Levels": table.shape[0], "Chi2": np.nan, "p_value": np.nan,
           "Cramers_V": np.nan, "Min_expected_count": np.nan}
    if table.shape[0] < 2 or table.shape[1] < 2:
        print(f"\n===== Bivariate: {col} vs {TARGET} - skipped (needs >= 2 levels) =====")
        return row

    chi2, p, _, expected = chi2_contingency(table)
    row.update({"Chi2": chi2, "p_value": p, "Cramers_V": cramers_v(table),
                "Min_expected_count": expected.min()})
    rate = (table[1] / table.sum(axis=1) * 100) if 1 in table.columns else pd.Series(0.0, index=table.index)
    print(f"\n===== Bivariate: {col} vs {TARGET} =====")
    print(pd.DataFrame({"n": table.sum(axis=1), "accident_rate_%": rate.round(2)}).to_string())
    print(f"Chi-square={chi2:.2f}, p={p:.4g}, Cramer's V={row['Cramers_V']:.3f}")
    if expected.min() < 5:
        print("⚠️ Some expected counts < 5 - chi-square result is unreliable; merge rare levels.")

    overall = df[TARGET].mean() * 100
    plot_rate = rate.sort_values(ascending=False).head(MAX_LEVELS_PLOTTED)
    plt.figure(figsize=(8, 4.5))
    plot_rate.rename(index=str).plot(kind="bar", color="#E4572E")
    plt.axhline(overall, color="black", linestyle="--", label=f"overall {overall:.2f}%")
    plt.ylabel("Accident rate (%)")
    plt.title(f"Accident rate by {col}")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    save_fig(f"cat_bi_{col}")
    return row


def leakage_check(df):
    present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if not present:
        return
    print("\n" + "=" * 70)
    print("LEAKAGE CHECK - these columns only exist after an accident happens.")
    print("A strong association below is EXPECTED and is NOT a usable predictor.")
    print("=" * 70)
    for col in present:
        if df[col].dtype.kind in "biufc" and df[col].nunique() > 20:
            print(f"{col}: numeric, {df[col].notna().mean() * 100:.1f}% filled "
                  f"(filled only for accident rows -> leakage)" if df[col].isna().any() else f"{col}: numeric")
            continue
        table = pd.crosstab(df[col].fillna("missing"), df[TARGET])
        print(f"{col}: Cramer's V with target = {cramers_v(table):.3f}")


def redundancy_checks(df):
    if {"is_peak", "peak"} <= set(df.columns):
        table = pd.crosstab(df["is_peak"], df["peak"])
        v = cramers_v(table)
        print(f"\nis_peak vs peak: Cramer's V = {v:.3f}"
              + ("  -> near duplicates, keep only one" if v > 0.95 else ""))
        print(table.to_string())


# ------------------------------- main -------------------------------
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {DATA_PATH}")
    df[TARGET] = df[TARGET].astype(int)   # the CSV may store True/False

    basic_overview(df)
    plot_target_balance(df)

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    numeric_univariate(df, num_cols)
    predictor_num_cols = [c for c in num_cols if c not in LEAKAGE_COLUMNS]
    numeric_vs_target(df, predictor_num_cols)

    cat_rows = []
    for col in get_categorical_columns(df):
        univariate_categorical(df, col)
        cat_rows.append(bivariate_categorical(df, col))
    if cat_rows:
        cat_summary = pd.DataFrame(cat_rows).sort_values("Cramers_V", ascending=False).round(4)
        cat_summary.to_excel(os.path.join(OUTPUT_DIR, "CatVsTarget_AccidentOccurred.xlsx"), index=False)
        print("\n### Categorical features vs target (sorted by Cramer's V) ###")
        print(cat_summary.to_string(index=False))

    redundancy_checks(df)
    leakage_check(df)
    print(f"\n✅ EDA complete. Plots and Excel summaries are in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
