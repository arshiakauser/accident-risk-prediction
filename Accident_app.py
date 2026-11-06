# -*- coding: utf-8 -*-

import os
import uuid
import warnings
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from preprocessing8 import (POST_EVENT_COLUMNS, RISK_LEVELS, get_missing_input_columns,
                            recommend_action, risk_level_from_proba, transform_test)

warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

MODEL_PATH = "best_model.pkl"
PREPROCESSING_PATH = "preprocessing_artifacts.pkl"
MAX_STYLED_ROWS = 1000

# The PostgreSQL upload is for private use. so the fields stay hidden unless you set the environment variable ENABLE_DB_UPLOAD=1.
ENABLE_DB_UPLOAD = os.getenv("ENABLE_DB_UPLOAD", "0") == "1"

COLOR_MAP = {
    "Low": [0, 200, 0],
    "Medium": [255, 165, 0],
    "High": [255, 0, 0],
    "Critical": [128, 0, 128],
}
RADIUS_MAP = {"Low": 100, "Medium": 200, "High": 300, "Critical": 400}


# ------------------- Artifacts -------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path, model_mtime, pre_path, pre_mtime):
    """mtime arguments are only there so the cache refreshes when the files are retrained."""
    return joblib.load(model_path), joblib.load(pre_path)


# ------------------- Scoring -------------------
def score_dataframe(new_data: pd.DataFrame, model_artifact: dict, preproc: dict) -> pd.DataFrame:
    """Return the uploaded rows plus prediction, probability, risk level and action."""
    clean = new_data.drop(columns=POST_EVENT_COLUMNS, errors="ignore")

    missing = get_missing_input_columns(clean, preproc["feature_engineer"])
    if missing:
        raise ValueError("The uploaded file is missing required columns: " + ", ".join(missing))

    X_new = transform_test(clean, preproc["feature_engineer"], preproc["preprocessor"],
                           preproc["selected_features"], preproc["train_columns"])

    threshold = model_artifact.get("threshold", 0.5)
    proba = model_artifact["pipeline"].predict_proba(X_new)[:, 1]
    levels = risk_level_from_proba(proba, threshold)

    results = clean.copy()
    results["Predicted_Accident"] = (proba >= threshold).astype(int)
    results["Accident_Probability"] = proba.round(4)
    results["Risk_Level"] = levels
    results["Recommended_Action"] = [recommend_action(level) for level in levels]
    return results


# ------------------- Database -------------------
def upload_to_postgres(results, user, password, host, port, database, table):
    url = URL.create("postgresql+psycopg2", username=user, password=password,
                     host=host, port=int(port), database=database)
    engine = create_engine(url)
    export = results.copy()
    export["run_id"] = uuid.uuid4().hex[:12]
    export["predicted_at"] = datetime.now()
    export.to_sql(table, engine, if_exists="append", index=False)
    engine.dispose()


def location_summary(results: pd.DataFrame) -> pd.DataFrame:
    """One row per map point (rows sharing a location are merged) so markers never hide each other.
    Colour = the highest risk level seen at that point."""
    df = results.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        return df
    rank = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
    df["_rank"] = df["Risk_Level"].map(rank).fillna(0)
    df["_flag"] = df["Risk_Level"].isin(["High", "Critical"]).astype(int)
    df["_lat"], df["_lon"] = df["latitude"].round(5), df["longitude"].round(5)
    out = (df.groupby(["_lat", "_lon"], as_index=False)
             .agg(rank=("_rank", "max"), max_prob=("Accident_Probability", "max"),
                  rows=("Risk_Level", "size"), flagged_rows=("_flag", "sum")))
    out["Risk_Level"] = out["rank"].astype(int).map(dict(enumerate(RISK_LEVELS)))
    out["latitude"], out["longitude"] = out["_lat"], out["_lon"]
    out["max_prob"] = out["max_prob"].round(3)
    return out.drop(columns=["rank", "_lat", "_lon"])


# ------------------- Rendering -------------------
def legend_html(levels, size=20):
    html = "<div style='display:flex;gap:20px;margin-bottom:10px;'>"
    for level in levels:
        r, g, b = COLOR_MAP[level]
        html += (f"<div style='display:flex;align-items:center;'><div style='width:{size}px;height:{size}px;"
                 f"background-color:rgb({r},{g},{b});margin-right:5px;'></div>{level}</div>")
    return html + "</div>"


def render_results(results: pd.DataFrame, threshold: float):
    present_levels = [lvl for lvl in RISK_LEVELS if lvl in set(results["Risk_Level"])]

    # ---- table ----
    st.write("### 🧾 Risk Predictions")
    st.markdown(legend_html(present_levels), unsafe_allow_html=True)

    def color_rows(row):
        r, g, b = COLOR_MAP.get(row["Risk_Level"], [255, 255, 255])
        return [f"background-color: rgba({r}, {g}, {b}, 0.3);"] * len(row)

    shown = results.head(MAX_STYLED_ROWS)
    st.dataframe(shown.style.apply(color_rows, axis=1), use_container_width=True)
    if len(results) > MAX_STYLED_ROWS:
        st.caption(f"Showing the first {MAX_STYLED_ROWS:,} of {len(results):,} rows. "
                   "The download contains all rows.")

    st.download_button("📥 Download Predictions as CSV",
                       data=results.to_csv(index=False).encode("utf-8"),
                       file_name="accident_predictions.csv", mime="text/csv")

    # ---- metrics ----
    st.markdown("### 📊 Summary Metrics")
    flagged = results["Risk_Level"].isin(["High", "Critical"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows scored", f"{len(results):,}")
    c2.metric("Flagged High / Critical", f"{flagged.mean() * 100:.2f}%")
    c3.metric("Critical rows", f"{(results['Risk_Level'] == 'Critical').sum():,}")
    c4.metric("Decision threshold", f"{threshold:.3f}")

    # ---- charts ----
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.write("### 📈 Risk Level Distribution")
        fig = px.histogram(
            results, x="Risk_Level", color="Risk_Level", title="Risk Category Counts",
            category_orders={"Risk_Level": RISK_LEVELS},
            color_discrete_map={k: f"rgb({v[0]},{v[1]},{v[2]})" for k, v in COLOR_MAP.items()})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if {"latitude", "longitude"} <= set(results.columns):
            st.write("### 🗺️ Accident Risk Map")
            map_df = location_summary(results)  # separate frame: never sent to the database
            if map_df.empty:
                st.info("ℹ️ No valid latitude/longitude values to plot.")
            else:
                map_df["color"] = map_df["Risk_Level"].apply(lambda x: COLOR_MAP.get(x, [0, 0, 0]))
                map_df["radius"] = map_df["Risk_Level"].map(RADIUS_MAP).fillna(100)
                st.markdown(legend_html(present_levels, size=15), unsafe_allow_html=True)
                st.caption(f"One marker per location ({len(map_df)} points); colour shows the highest risk "
                           "level at that location. Hover for details.")
                layers = [pdk.Layer("ScatterplotLayer", data=map_df, get_position="[longitude, latitude]",
                                    get_color="color", get_radius="radius", pickable=True, opacity=0.7)]
                st.pydeck_chart(pdk.Deck(
                    map_provider="carto", map_style="light",  # no Mapbox token needed
                    initial_view_state=pdk.ViewState(latitude=map_df["latitude"].mean(),
                                                     longitude=map_df["longitude"].mean(),
                                                     zoom=11, pitch=0),
                    layers=layers,
                    tooltip={"html": "<b>Highest risk:</b> {Risk_Level}<br/>"
                                     "<b>Max score:</b> {max_prob}<br/>"
                                     "<b>Rows here:</b> {rows} ({flagged_rows} flagged)<br/>"
                                     "<b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
                             "style": {"backgroundColor": "white", "color": "black"}}))
        else:
            st.info("ℹ️ Latitude/Longitude columns not found. Map cannot be displayed.")


# ------------------- App -------------------
def main():
    st.set_page_config(page_title="Accident Risk Prediction", layout="wide")
    st.title("🚦 Smart Accident Risk Classification")
    st.markdown("""
    This app predicts **accident risk levels (Low–Critical)** with a **threshold model**
    and suggests preventive traffic management actions.
    """)

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSING_PATH)):
        st.error(f"Model files not found. Expected `{MODEL_PATH}` and `{PREPROCESSING_PATH}` "
                 "next to this app. Run the training script first.")
        st.stop()
    model_artifact, preproc = load_artifacts(MODEL_PATH, os.path.getmtime(MODEL_PATH),
                                             PREPROCESSING_PATH, os.path.getmtime(PREPROCESSING_PATH))
    threshold = model_artifact.get("threshold", 0.5)

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Settings")
    #st.sidebar.caption(f"Model: **{model_artifact.get('model_name', 'unknown')}** · "
                       #f"threshold {threshold:.3f}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload intersection/traffic data (CSV)", type=["csv"])

    db_user = db_pw = db_name = None
    db_host, db_port, table_name = "localhost", "5432", "accident_predictions"
    if ENABLE_DB_UPLOAD:
        st.sidebar.markdown("---")
        st.sidebar.subheader("PostgreSQL Connection (Optional)")
        db_user = st.sidebar.text_input("User")
        db_pw = st.sidebar.text_input("Password", type="password")
        db_host = st.sidebar.text_input("Host", "localhost")
        db_port = st.sidebar.text_input("Port", "5432")
        db_name = st.sidebar.text_input("Database")
        table_name = st.sidebar.text_input("Target Table", "accident_predictions")
    predict_btn = st.sidebar.button("🔮 Predict & Upload" if ENABLE_DB_UPLOAD else "🔮 Predict")

    # ---------------- Predict ----------------
    if predict_btn:
        if uploaded_file is None:
            st.warning("Upload a CSV file first.")
        else:
            try:
                new_data = pd.read_csv(uploaded_file)
                with st.spinner("🔍 Running risk predictions..."):
                    results = score_dataframe(new_data, model_artifact, preproc)
                st.session_state["results"] = results
                st.session_state["threshold"] = threshold
                st.session_state["db_status"] = None

                if not ENABLE_DB_UPLOAD:
                    st.session_state["db_status"] = None
                elif db_user and db_pw and db_name:
                    try:
                        upload_to_postgres(results, db_user, db_pw, db_host, db_port, db_name, table_name)
                        st.session_state["db_status"] = ("success",
                            f"✅ Results uploaded to `{table_name}` in database `{db_name}`.")
                    except Exception as exc:
                        st.session_state["db_status"] = ("error", f"❌ Upload failed: {exc}")
                else:
                    st.session_state["db_status"] = ("info", "💡 Enter database credentials to enable upload.")
            except ValueError as exc:  # missing columns etc.
                st.session_state.pop("results", None)
                st.error(f"❌ {exc}")
            except Exception as exc:
                st.session_state.pop("results", None)
                st.error(f"❌ Prediction failed: {exc}")

    # ---------------- Show (survives reruns such as clicking Download) ----------------
    results = st.session_state.get("results")
    if results is None:
        st.info("📂 Upload a CSV file in the sidebar and click the **Predict** button to begin.")
        return

    render_results(results, st.session_state.get("threshold", threshold))
    status = st.session_state.get("db_status")
    if status:
        getattr(st, status[0])(status[1])


if __name__ == "__main__":
    main()
