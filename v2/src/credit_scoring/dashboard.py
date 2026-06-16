"""Credit scoring dashboard. Offline mode uses local model; online mode calls serving API."""

import os
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests, shap, streamlit as st, mlflow.sklearn

DASHBOARD_MODE = os.environ.get("DASHBOARD_MODE", "offline")
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MODEL_DIR = os.environ.get("MODEL_DIR", "production_model")
ANALYSIS_DATA_PATH = os.environ.get("ANALYSIS_DATA_PATH", "data_for_analysis.csv")
GLOBAL_IMPORTANCE_PATH = os.environ.get("GLOBAL_IMPORTANCE_PATH", "global_feature_importance.json")
BEST_THRESHOLD = float(os.environ.get("BEST_THRESHOLD", "0.13"))

st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")


@st.cache_resource
def load_model() -> Any:
    return mlflow.sklearn.load_model(MODEL_DIR)


@st.cache_data
def load_analysis_data() -> pd.DataFrame:
    if os.path.exists(ANALYSIS_DATA_PATH):
        return pd.read_csv(ANALYSIS_DATA_PATH)
    demo = os.path.join(MODEL_DIR, "input_example.json")
    if os.path.exists(demo):
        return pd.read_json(demo, orient="split")
    st.error("Analysis data not found.")
    return pd.DataFrame()


@st.cache_data
def load_global_importance() -> pd.DataFrame:
    if os.path.exists(GLOBAL_IMPORTANCE_PATH):
        return pd.read_json(GLOBAL_IMPORTANCE_PATH)
    st.warning("Global importance file not found.")
    return pd.DataFrame(columns=["feature", "importance"])


analysis_data = load_analysis_data()
global_importance_df = load_global_importance()
EXPECTED_FEATURES = list(analysis_data.drop(columns=["TARGET"], errors="ignore").columns)
if DASHBOARD_MODE == "offline":
    model = load_model()

st.sidebar.title("Client & Feature Controls")
KNOWN_CLIENTS = {f"Client_ID_{i}": row.to_dict() for i, row in analysis_data.iterrows()}
cid = st.sidebar.selectbox("Select Client", [""] + list(KNOWN_CLIENTS.keys()))
prefill = KNOWN_CLIENTS[cid] if cid else analysis_data.drop(columns=["TARGET"], errors="ignore").median().to_dict()

st.sidebar.header("Feature Input")
with st.sidebar.expander("Adjust Features", expanded=True):
    cd: dict[str, Any] = {}
    for feat in EXPECTED_FEATURES:
        s = analysis_data[feat]
        d = prefill.get(feat, s.median())
        if s.min() == s.max():
            st.text_input(feat, value=str(d), disabled=True); cd[feat] = d
        elif pd.api.types.is_numeric_dtype(s.dtype):
            cd[feat] = st.slider(feat, float(s.min()), float(s.max()), float(d), key=f"sl_{feat}")
        else:
            opts = sorted(s.unique())
            cd[feat] = st.radio(feat, opts, index=opts.index(d) if d in opts else 0, key=f"rd_{feat}")

if st.sidebar.button("Analyze Client", type="primary", use_container_width=True):
    with st.spinner("Analyzing..."):
        if DASHBOARD_MODE == "online":
            resp = requests.post(
                f"{API_URL}/predict?client_id={cid or 'custom'}", json={"features": cd}
            )
            if resp.status_code != 200:
                st.error(f"API error: {resp.json().get('detail', resp.text)}"); st.stop()
            st.session_state.prob = resp.json()["probability"]
        else:
            input_df = pd.DataFrame([cd], columns=EXPECTED_FEATURES)
            st.session_state.prob = model.predict_proba(input_df)[0, 1]
            pp = model.steps[0][1]
            clf = model.steps[1][1]
            proc = pp.transform(input_df)
            exp = shap.TreeExplainer(clf)(proc)
            st.session_state.shap_exp = shap.Explanation(
                values=exp.values[:, :, 1], base_values=exp.base_values[:, 1],
                data=exp.data, feature_names=pp.get_feature_names_out(input_features=input_df.columns),
            )
        st.session_state.cd, st.session_state.cid, st.session_state.analysis = cd, cid, True

st.title("Credit Scoring & Risk Analysis Dashboard")
if not st.session_state.get("analysis", False):
    st.info("Select a client or adjust features and click 'Analyze Client'."); st.stop()

prob = st.session_state.prob
pred = 1 if prob >= BEST_THRESHOLD else 0
status = "High Risk (Default)" if pred == 1 else "Low Risk (No Default)"
st.header(f"Analysis for: **{st.session_state.cid or 'Custom Client'}**")
st.subheader("Risk Assessment")
st.metric("Status", status)
st.progress(prob, text=f"Default Probability: {prob:.2%}")
st.caption(f"Threshold: {BEST_THRESHOLD:.0%}")

if DASHBOARD_MODE == "offline":
    st.markdown("---")
    st.subheader("Feature Contribution to Prediction")
    tab1, tab2 = st.tabs(["Local Importance (This Client)", "Global Importance (All Clients)"])
    with tab1:
        try:
            exp = st.session_state.shap_exp[0]
            imp = pd.DataFrame({"f": exp.feature_names, "v": exp.values}).assign(
                a=lambda df: df["v"].abs()
            ).sort_values("a", ascending=False)
            top = imp.head(7)
            oth = imp.iloc[7:]
            if not oth.empty:
                top = pd.concat([top, pd.DataFrame([{"f": f"{len(oth)} Other Features", "v": oth["v"].sum()}])], ignore_index=True)
            top = top.sort_values("v", ascending=False)
            bv, fv = float(exp.base_values), float(exp.base_values + np.sum(exp.values))
            yv, xl = top["v"].tolist(), top["f"].tolist()
            fig = go.Figure(go.Waterfall(
                orientation="v", measure=["absolute"] + ["relative"] * len(yv) + ["total"],
                x=["Average Prediction"] + xl + ["Final Prediction"],
                y=[bv] + yv + [fv],
                text=[f"{v:.3f}" for v in [bv] + yv + [fv]],
                textposition="outside", connector={"line": {"color": "rgb(63,63,63)"}},
                increasing={"marker": {"color": "#d62728"}},
                decreasing={"marker": {"color": "#1f77b4"}},
                totals={"marker": {"color": "#2ca02c"}},
            ))
            fig.update_layout(height=600, yaxis_title="Probability Impact")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"SHAP plot error: {e}")
    with tab2:
        if not global_importance_df.empty:
            gi = global_importance_df.head(20).sort_values("importance", ascending=True)
            fig = px.bar(gi, x="importance", y="feature", orientation="h",
                         title="Top 20 Global Features", text="importance")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Global importance data not available.")
    st.markdown("---")

st.header("Client Data Analysis")
tab3, tab4 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

with tab3:
    c1, c2 = st.columns([3, 1])
    with c1:
        fu = st.selectbox("Select a feature", EXPECTED_FEATURES, index=0)
    with c2:
        flt = st.selectbox("Filter by Target", ["All", 0, 1], key="uf")
    pu = analysis_data.copy()
    if flt != "All":
        pu = pu[pu["TARGET"] == int(flt)]
    cu = "TARGET" if "TARGET" in pu.columns and flt == "All" else None
    if cu:
        pu = pu.assign(TARGET=pu["TARGET"].astype(str))
    fig = px.histogram(pu, x=fu, color=cu, color_discrete_map={"0": "blue", "1": "red"})
    v = st.session_state.cd.get(fu)
    if v is not None:
        fig.add_vline(x=v, line_width=3, line_dash="dash", line_color="yellow", annotation_text="Current Client")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        xf = st.selectbox("Select X-axis feature", EXPECTED_FEATURES, index=1, key="bx")
    with c2:
        yf = st.selectbox("Select Y-axis feature", EXPECTED_FEATURES, index=2, key="by")
    with c3:
        flb = st.selectbox("Filter by Target", ["All", 0, 1], key="bf")
    pb = analysis_data.copy()
    if flb != "All":
        pb = pb[pb["TARGET"] == int(flb)]
    xc, yc = analysis_data[xf].nunique() < 10, analysis_data[yf].nunique() < 10
    if (xc and not yc) or (not xc and yc):
        cb = None
        if flb == "All":
            pb = pb.assign(Status=pb["TARGET"].map({0: "No Default", 1: "Default"})); cb = "Status"
        fig = px.box(pb, x=xf, y=yf, color=cb, color_discrete_map={"No Default": "blue", "Default": "red"})
    else:
        if flb == "All":
            fig = go.Figure()
            for lbl, tv, col in [("No Default (0)", 0, "blue"), ("Default (1)", 1, "red")]:
                sub = pb[pb["TARGET"] == tv]
                fig.add_trace(go.Scatter(x=sub[xf], y=sub[yf], mode="markers", marker=dict(color=col), name=lbl))
            fig.update_layout(xaxis_title=xf, yaxis_title=yf, legend_title_text="Status")
        else:
            fig = px.scatter(pb, x=xf, y=yf)
            fig.update_traces(marker=dict(color="blue" if flb == 0 else "red", size=7))
    cx, cy = st.session_state.cd.get(xf), st.session_state.cd.get(yf)
    if cx is not None and cy is not None:
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                      marker=dict(color="yellow", size=15, symbol="star", line=dict(color="black", width=1)),
                      name="Current Client"))
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(f"Dashboard mode: {DASHBOARD_MODE}")
