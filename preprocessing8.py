# -*- coding: utf-8 -*-

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

try:  # SHAP is optional: fall back to impurity importance if it is not installed
    import shap
except ImportError:  # pragma: no cover
    shap = None

# ----------------------------------------------------------------------------
# Shared constants (used by the training script and the Streamlit app)
# ----------------------------------------------------------------------------
TARGET = "accident_occurred"

# Columns that only exist AFTER an accident happens -> never allowed as inputs.
POST_EVENT_COLUMNS = [TARGET, "vehicles_involved", "veh_count_at_accident", "severity", "cause"]

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

# Category values used to build interaction features. They are matched case-insensitively.
# CHECK THEM against df["weather"].unique(), df["lighting"].unique(), df["enforcement_level"].unique();
# FeatureEngineer.fit() warns if one of them never occurs in the training data.
WEATHER_LEVELS = ("rain", "thunderstorm", "clear")
LIGHTING_LEVELS = ("night", "day")
HIGH_ENFORCEMENT_LEVEL = "high"

# Identifier columns (this dataset uses location_id; older exports used intersection_id).
ID_COLUMNS = ("intersection_id", "location_id")
# Columns that duplicate another column (in your data `peak` == `is_peak` in 99.9% of rows).
DUPLICATE_COLUMNS = ("peak",)

_DAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _safe_div(num, den):
    """num / den, but 0 where den == 0. Missing values stay missing (imputed later)."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den.where(den != 0)
    return out.where(~(den == 0), 0.0)


def _norm(series):
    return series.astype(str).str.strip().str.lower()


def _to_flag(series):
    """True/False, 0/1, yes/no -> 0/1 integers (unknown -> 0)."""
    if series.dtype == bool:
        return series.astype(int)
    num = pd.to_numeric(series, errors="coerce")
    txt = _norm(series).isin({"true", "yes", "y", "t"}).astype(float)
    return num.fillna(txt).fillna(0).clip(0, 1).astype(int)


def _day_to_int(series):
    """Monday..Sunday (or 0..6) -> 0..6, fixed mapping, unknown -> -1."""
    num = pd.to_numeric(series, errors="coerce")
    by_name = _norm(series).str[:3].map(_DAY_MAP)
    return num.fillna(by_name).fillna(-1).astype(int)


def split_numeric_categorical(df):
    """(numeric_cols, categorical_cols). Works with pandas 2 (object) and pandas 3 (str) text columns."""
    num_cols, cat_cols = [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            num_cols.append(c)
        elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
              or isinstance(s.dtype, pd.CategoricalDtype)):
            cat_cols.append(c)
    return num_cols, cat_cols


def sanitize_feature_names(columns):
    """Make names safe for LightGBM / XGBoost (letters, digits, underscore only)."""
    return [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in columns]


# ----------------------------------------------------------------------------
# 1) Outlier capping
# ----------------------------------------------------------------------------
class OutlierCapper(BaseEstimator, TransformerMixin):
    """Clips each column to its [lower_q, upper_q] training percentiles.

    Columns with <= 2 distinct values (0/1 flags) are left untouched, otherwise a flag that
    is 1 in fewer than 1% of rows would be capped to 0 and its signal would disappear.
    """

    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        lower = np.nanquantile(X, self.lower_q, axis=0)
        upper = np.nanquantile(X, self.upper_q, axis=0)
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.unique(col[~np.isnan(col)]).size <= 2:
                lower[j], upper[j] = -np.inf, np.inf
        self.lower_, self.upper_ = lower, upper
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ----------------------------------------------------------------------------
# 2) Feature engineering
# ----------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates ratio, interaction, cluster and code features.

    Everything that depends on data (intersection codes, KMeans, medians) is learned in
    fit() and only re-used in transform(), so one row scored alone gets exactly the same
    features as the same row inside a big file.
    """

    CLUSTER_FEATURES = [
        "vehicle_count_per_hr", "avg_speed_kmph", "cycle_time_s",
        "lane_count", "speed_limit_kmph", "violations_count", "blackspot_score",
    ]

    def __init__(self, n_clusters=5, random_state=42, use_id_codes=False,
                 drop_columns=("timestamp", "intersection_id", "location_id", "peak"),
                 optional_columns=("timestamp", "intersection_id", "location_id", "peak")):
        self.n_clusters = n_clusters
        self.random_state = random_state
        # False (default): id columns are dropped. lat/long/blackspot_score already identify a site, and an
        # arbitrary integer code for an id is not an ordered quantity. True: add <id>_factorized codes.
        self.use_id_codes = use_id_codes
        self.drop_columns = drop_columns          # removed from the model input after features are built
        self.optional_columns = optional_columns  # may be missing in new data (unless use_id_codes needs them)

    # -- fit -------------------------------------------------------------
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.input_columns_ = X.columns.tolist()

        self.id_maps_ = {}
        if self.use_id_codes:
            for col in ID_COLUMNS:
                if col in X.columns:
                    ids = pd.Series(X[col].dropna().unique()).astype(object)
                    self.id_maps_[col] = {v: i for i, v in enumerate(sorted(ids, key=str))}

        # columns with a single value (e.g. state / season in a one-week extract) carry no information
        self.constant_columns_ = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if self.constant_columns_:
            warnings.warn(f"FeatureEngineer: constant columns will be dropped: {self.constant_columns_}")

        self.cluster_columns_ = [c for c in self.CLUSTER_FEATURES if c in X.columns]
        self.scaler_, self.kmeans_, self.cluster_medians_ = None, None, None
        if self.cluster_columns_:
            num = X[self.cluster_columns_].apply(pd.to_numeric, errors="coerce")
            self.cluster_medians_ = num.median()
            filled = num.fillna(self.cluster_medians_).to_numpy()
            self.scaler_ = StandardScaler().fit(filled)
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                                  n_init=10).fit(self.scaler_.transform(filled))

        # Warn loudly if the interaction categories do not exist in the data
        for col, levels in (("weather", WEATHER_LEVELS), ("lighting", LIGHTING_LEVELS),
                            ("enforcement_level", (HIGH_ENFORCEMENT_LEVEL,))):
            if col in X.columns:
                present = set(_norm(X[col]).unique())
                missing = [v for v in levels if v not in present]
                if missing:
                    warnings.warn(f"FeatureEngineer: values {missing} never occur in column '{col}' "
                                  f"(found {sorted(present)}). Update the *_LEVELS constants at the "
                                  f"top of preprocessing8.py so the interaction features are created.")
        return self

    # -- transform -------------------------------------------------------
    def transform(self, X):
        check_is_fitted(self, "input_columns_")
        X_new = pd.DataFrame(X).copy()
        cols = set(X_new.columns)

        # Signal ratios
        if {"green_duration_s", "cycle_time_s"} <= cols:
            X_new["green_ratio"] = _safe_div(X_new["green_duration_s"], X_new["cycle_time_s"])
        if {"red_duration_s", "cycle_time_s"} <= cols:
            X_new["red_ratio"] = _safe_div(X_new["red_duration_s"], X_new["cycle_time_s"])
        if {"yellow_duration_s", "cycle_time_s"} <= cols:
            X_new["yellow_ratio"] = _safe_div(X_new["yellow_duration_s"], X_new["cycle_time_s"])
        if {"red_duration_s", "green_duration_s"} <= cols:
            X_new["red_to_green_ratio"] = _safe_div(X_new["red_duration_s"], X_new["green_duration_s"])

        # Congestion & overspeeding
        if {"vehicle_count_per_hr", "lane_count"} <= cols:
            X_new["congestion_density"] = _safe_div(X_new["vehicle_count_per_hr"], X_new["lane_count"])
        if {"avg_speed_kmph", "speed_limit_kmph"} <= cols:
            X_new["overspeeding_index"] = _safe_div(X_new["avg_speed_kmph"], X_new["speed_limit_kmph"])
        if {"violations_count", "vehicle_count_per_hr"} <= cols:
            X_new["violations_per_vehicle"] = _safe_div(X_new["violations_count"], X_new["vehicle_count_per_hr"])

        # Peak hours (new name so the raw is_peak / peak columns are not overwritten)
        if "hour_of_day" in cols:
            h = pd.to_numeric(X_new["hour_of_day"], errors="coerce")
            X_new["peak_hour_flag"] = (((h >= 7) & (h <= 9)) | ((h >= 17) & (h <= 20))).astype(int)

        # Interactions between engineered features
        if "overspeeding_index" in X_new.columns and "peak_hour_flag" in X_new.columns:
            X_new["overspeed_peak"] = X_new["overspeeding_index"] * X_new["peak_hour_flag"]
        if "red_ratio" in X_new.columns and "lane_count" in cols:
            X_new["red_lane_interaction"] = X_new["red_ratio"] * pd.to_numeric(X_new["lane_count"], errors="coerce")

        # Enforcement x overspeeding (built from the RAW column)
        if "enforcement_level" in cols:
            X_new["enforcement_high"] = (_norm(X_new["enforcement_level"]) == HIGH_ENFORCEMENT_LEVEL).astype(int)
            if "overspeeding_index" in X_new.columns:
                X_new["enf_overspeed_interaction"] = X_new["enforcement_high"] * X_new["overspeeding_index"]

        # Weather x lighting (built from the RAW columns)
        if {"weather", "lighting"} <= cols:
            w, l = _norm(X_new["weather"]), _norm(X_new["lighting"])
            for wv in WEATHER_LEVELS:
                for lv in LIGHTING_LEVELS:
                    X_new[f"{wv}_{lv}"] = ((w == wv) & (l == lv)).astype(int)

        # KMeans cluster (categorical label, not an ordered number)
        if self.kmeans_ is not None:
            missing = [c for c in self.cluster_columns_ if c not in cols]
            if missing:
                raise ValueError(f"Columns needed for kmeans_cluster are missing: {missing}")
            num = X_new[self.cluster_columns_].apply(pd.to_numeric, errors="coerce")
            filled = num.fillna(self.cluster_medians_).to_numpy()
            labels = self.kmeans_.predict(self.scaler_.transform(filled))
            X_new["kmeans_cluster"] = ["cluster_" + str(k) for k in labels]

        # Codes with a FIXED mapping learned in fit()
        for col, mapping in self.id_maps_.items():
            if col in cols:
                X_new[f"{col}_factorized"] = X_new[col].map(mapping).fillna(-1).astype(int)
        if "day_of_week" in cols and not pd.api.types.is_numeric_dtype(X_new["day_of_week"]):
            X_new["day_factorized"] = _day_to_int(X_new["day_of_week"])
        if "is_holiday" in cols:
            X_new["is_holiday"] = _to_flag(X_new["is_holiday"])

        # Remove columns that must not reach the model (ids, raw timestamps, duplicates, constants)
        to_drop = set(self.drop_columns) | set(self.constant_columns_)
        X_new = X_new.drop(columns=[c for c in X_new.columns if c in to_drop])

        for c in X_new.columns:
            s_c = X_new[c]
            if pd.api.types.is_bool_dtype(s_c):          # booleans -> 0/1 (numeric everywhere)
                X_new[c] = s_c.astype(int)
            elif not pd.api.types.is_numeric_dtype(s_c):  # text / mixed -> uniform strings, NaN kept
                X_new[c] = s_c.astype(object).where(s_c.isna(), s_c.astype(str))
        return X_new


# ----------------------------------------------------------------------------
# 3) Helpers shared by training, prediction and the app
# ----------------------------------------------------------------------------
def get_output_feature_names(preprocessor):
    """Sanitized output names of a fitted preprocessor. ONE function = train and test always agree."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(trans.named_steps["imputer"].get_feature_names_out(list(cols)).tolist())
        elif name == "cat":
            names.extend(trans.named_steps["onehot"].get_feature_names_out(list(cols)).tolist())
    names = sanitize_feature_names(names)
    duplicates = pd.Series(names)[pd.Series(names).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate feature names after sanitizing: {duplicates}")
    return names


def get_missing_input_columns(new_data, feature_engineer):
    """Raw columns the model needs that are not present in new_data."""
    optional = set(getattr(feature_engineer, "optional_columns", ()) or ())
    optional -= set(getattr(feature_engineer, "id_maps_", {}))        # ids are required when codes are used
    optional |= set(getattr(feature_engineer, "constant_columns_", []))
    expected = [c for c in feature_engineer.input_columns_ if c not in optional]
    return [c for c in expected if c not in new_data.columns]


def risk_level_from_proba(proba, threshold):
    """Risk bands anchored on the model's decision threshold.

    Low      : p < threshold / 2
    Medium   : threshold / 2 <= p < threshold
    High     : threshold <= p < halfway between threshold and 1
    Critical : p >= halfway between threshold and 1
    => "High" or "Critical" is exactly "Predicted accident = 1". The model is trained on
    resampled data, so probabilities are NOT calibrated; fixed 0.25/0.5/0.75 bands would
    contradict the prediction.
    """
    p = np.asarray(proba, dtype=float)
    t = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    critical_cut = t + 0.5 * (1 - t)
    return np.select([p < 0.5 * t, p < t, p < critical_cut], ["Low", "Medium", "High"], default="Critical")


_ACTIONS = {
    "Low": "✅ Low Risk - Normal monitoring",
    "Medium": "⚠️ Medium Risk - Increase patrolling",
    "High": "🚧 High Risk - Adjust signals, deploy signage",
    "Critical": "🚨 Critical Risk - Reroute or alert emergency units",
}


def recommend_action(level):
    return _ACTIONS.get(str(level), "ℹ️ Unknown risk level")


# ----------------------------------------------------------------------------
# 4) Build the preprocessing pipeline (fit on TRAINING data only)
# ----------------------------------------------------------------------------
def _positive_class_shap(shap_values):
    """Return the (n_samples, n_features) SHAP array of the minority/positive class."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, -1] if arr.shape[2] == 2 else arr[-1]
    return arr


def _rank_features(X_df, y, random_state=42, shap_sample=5000):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                min_samples_leaf=2, n_jobs=-1, random_state=random_state)
    rf.fit(X_df, y)
    if shap is not None:
        sample = X_df.sample(min(len(X_df), shap_sample), random_state=random_state)
        values = _positive_class_shap(shap.TreeExplainer(rf).shap_values(sample))
        importance = np.abs(values).mean(axis=0)
    else:
        print("   (shap not installed - using Random Forest impurity importance instead)")
        importance = rf.feature_importances_
    return pd.Series(importance, index=X_df.columns).sort_values(ascending=False)


def build_preprocessing_pipeline_shap(X, y, n_features_select=25, shap_preselect=50,
                                      use_sfs=False, random_state=42, extra_drop_columns=()):
    """Fit feature engineering + preprocessing + feature selection on TRAINING data.

    Returns (X_selected, selected_features, feature_engineer, preprocessor) - same as before.
    use_sfs=True adds a forward SequentialFeatureSelector (average_precision) on the top
    `shap_preselect` features. It is slow; SHAP ranking alone is usually enough.
    extra_drop_columns: more raw columns to keep out of the model (they also become optional in new
    data), e.g. ("day_of_week",) when the training data covers only a few days.
    """
    print("\n🔧 Starting preprocessing pipeline...")
    print(f"   Input shape: {X.shape}")

    # 1) feature engineering FIRST, so engineered columns are part of the schema
    defaults = FeatureEngineer()
    extra = tuple(extra_drop_columns)
    feature_engineer = FeatureEngineer(
        n_clusters=5, random_state=random_state,
        drop_columns=tuple(defaults.drop_columns) + extra,
        optional_columns=tuple(defaults.optional_columns) + extra)
    X_fe = feature_engineer.fit_transform(X)
    print(f"   After feature engineering: {X_fe.shape}")

    num_cols, cat_cols = split_numeric_categorical(X_fe)
    ignored = [c for c in X_fe.columns if c not in num_cols + cat_cols]
    if ignored:
        print(f"   ⚠️ Ignoring columns with unsupported dtype: {ignored}")
    print(f"   Numeric columns: {len(num_cols)} | Categorical columns: {len(cat_cols)}")

    # 2) column transformer built from the post-engineering columns
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            # add_indicator: "missing" is informative here (no signal => no cycle time), so keep a 0/1 flag
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("capper", OutlierCapper()),
            ("scaler", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            # NaN means "no event / no signal" in this data, so it becomes its own category. "most_frequent"
            # would have relabelled ~98% empty quality-flag columns as e.g. 'veh_spike'.
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_pre = preprocessor.fit_transform(X_fe)
    feature_names = get_output_feature_names(preprocessor)
    X_pre_df = pd.DataFrame(X_pre, columns=feature_names, index=X.index).astype(float)
    print(f"   After preprocessing: {X_pre_df.shape}")

    # 3) feature selection
    print("\n   Ranking features (Random Forest + SHAP)...")
    ranking = _rank_features(X_pre_df, y, random_state=random_state)

    available = X_pre_df.shape[1]
    shap_preselect = min(shap_preselect, available)
    n_features_select = max(1, min(n_features_select, shap_preselect - 1 if use_sfs else shap_preselect))
    top_features = ranking.head(shap_preselect).index.tolist()

    if use_sfs:
        print(f"   Running Sequential Feature Selection ({shap_preselect} -> {n_features_select})...")
        sfs = SequentialFeatureSelector(
            RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                   min_samples_leaf=2, n_jobs=1, random_state=random_state),
            n_features_to_select=n_features_select,
            scoring="average_precision",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            direction="forward", n_jobs=-1,
        )
        sfs.fit(X_pre_df[top_features], y)
        selected_features = X_pre_df[top_features].columns[sfs.get_support()].tolist()
    else:
        selected_features = top_features[:n_features_select]

    print(f"\n✅ Feature selection complete: {len(selected_features)} of {available} features kept")
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"      {i}. {feat}")
    if len(selected_features) > 10:
        print(f"      ... and {len(selected_features) - 10} more")

    engineered = set(sanitize_feature_names(X_fe.columns)) - set(sanitize_feature_names(X.columns))
    n_eng = sum(any(f == e or f.startswith(e + "_") for e in engineered) for f in selected_features)
    print(f"   Engineered features among the selected ones: {n_eng}")

    return X_pre_df[selected_features], selected_features, feature_engineer, preprocessor


# ----------------------------------------------------------------------------
# 5) Transform new / test data
# ----------------------------------------------------------------------------
def transform_test(new_data, feature_engineer, preprocessor, selected_features, train_columns=None):
    """Apply the fitted feature engineering + preprocessing to new data.

    Raises ValueError when raw columns or expected features are missing instead of
    silently filling them with zeros (which would give wrong predictions without any error).
    """
    new_data = pd.DataFrame(new_data)

    missing_raw = get_missing_input_columns(new_data, feature_engineer)
    if missing_raw:
        raise ValueError(f"Input data is missing required columns: {missing_raw}")

    X_fe = feature_engineer.transform(new_data)

    needed = [c for name, _, cols in preprocessor.transformers_ if name in ("num", "cat") for c in cols]
    absent = [c for c in needed if c not in X_fe.columns]
    if absent:
        raise ValueError(f"Columns missing after feature engineering: {absent}")

    X_pre = preprocessor.transform(X_fe)
    names = get_output_feature_names(preprocessor)
    X_pre_df = pd.DataFrame(X_pre, columns=names, index=new_data.index)

    wanted = list(train_columns) if train_columns is not None else list(selected_features)
    missing_feats = [c for c in wanted if c not in X_pre_df.columns]
    if missing_feats:
        raise ValueError(f"Features expected by the model are missing: {missing_feats}")
    return X_pre_df[wanted]
