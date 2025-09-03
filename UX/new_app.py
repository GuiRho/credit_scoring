import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import shap
import matplotlib.pyplot as plt
import mlflow.sklearn
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Offline Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide"
)

# --- Path and Constant Definitions ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "production_model")
DEMO_DATA_PATH = os.path.join(MODEL_PATH, "input_example.json")
ANALYSIS_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_for_analysis.csv")
GLOBAL_IMPORTANCE_PATH = os.path.join(MODEL_PATH, "global_feature_importance.json")
BEST_THRESHOLD = 0.13

# --- Caching Functions for Performance ---
@st.cache_resource
def load_model():
    """Loads the full MLflow pipeline model."""
    return mlflow.sklearn.load_model(MODEL_PATH)

@st.cache_data
def load_demo_data():
    return pd.read_json(DEMO_DATA_PATH, orient="split")

@st.cache_data
def load_analysis_data():
    if os.path.exists(ANALYSIS_DATA_PATH):
        return pd.read_csv(ANALYSIS_DATA_PATH)
    else:
        st.warning(f"Warning: '{ANALYSIS_DATA_PATH}' not found. Using small demo dataset for plots.")
        return load_demo_data()

@st.cache_data
def load_global_importance():
    """Loads the global feature importance data calculated by analysis.py."""
    if os.path.exists(GLOBAL_IMPORTANCE_PATH):
        return pd.read_json(GLOBAL_IMPORTANCE_PATH)
    else:
        st.error(f"FATAL: Global importance file not found at '{GLOBAL_IMPORTANCE_PATH}'. Please run analysis.py first.")
        return pd.DataFrame(columns=['feature', 'importance'])

# --- Data and Model Loading ---
model = load_model()
analysis_data = load_analysis_data()
global_importance_df = load_global_importance()
EXPECTED_FEATURES = list(analysis_data.drop(columns=['TARGET']).columns)

# --- UI: Sidebar for Inputs ---
st.sidebar.title("Client & Feature Controls")
st.sidebar.markdown("---")

KNOWN_CLIENTS = {
    f"Client_ID_{analysis_data.index[i]}": analysis_data.iloc[i].to_dict()
    for i in range(len(analysis_data))
}
client_id = st.sidebar.selectbox("Select a Known Client (for Demo)", [""] + list(KNOWN_CLIENTS.keys()))

if client_id:
    prefill_data = KNOWN_CLIENTS[client_id]
else:
    prefill_data = analysis_data.drop(columns=['TARGET']).median().to_dict()

st.sidebar.header("Client Feature Input")
with st.sidebar.expander("Adjust Client Features", expanded=True):
    client_data = {}
    for feature in EXPECTED_FEATURES:
        series = analysis_data[feature]
        min_val, max_val, dtype = series.min(), series.max(), series.dtype
        default_val = prefill_data.get(feature, series.median())

        if min_val == max_val:
            st.text_input(f"{feature} (Constant)", value=default_val, disabled=True)
            client_data[feature] = default_val
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            client_data[feature] = st.slider(
                feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                key=f"slider_{feature}"
            )
        else:
            options = sorted(series.unique())
            try:
                default_index = options.index(default_val)
            except (ValueError, IndexError):
                default_index = 0
            client_data[feature] = st.radio(
                feature,
                options,
                index=default_index,
                key=f"radio_{feature}"
            )

if 'analysis_generated' not in st.session_state:
    st.session_state.analysis_generated = False

if st.sidebar.button("Analyze Client", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction and feature contributions..."):
        input_df = pd.DataFrame([client_data], columns=EXPECTED_FEATURES)
        st.session_state.prob = model.predict_proba(input_df)[0, 1]
        
        preprocessor = model.steps[0][1]
        classifier = model.steps[1][1]
        processed_input_df = preprocessor.transform(input_df)
        explainer = shap.TreeExplainer(classifier)
        shap_explanation_object = explainer(processed_input_df)
        
        st.session_state.shap_explanation_for_plot = shap.Explanation(
            values=shap_explanation_object.values[:, :, 1],
            base_values=shap_explanation_object.base_values[:, 1],
            data=shap_explanation_object.data,
            feature_names=preprocessor.get_feature_names_out(input_features=input_df.columns)
        )
        
        st.session_state.client_data_for_plots = client_data
        st.session_state.client_id_for_plots = client_id
        st.session_state.analysis_generated = True

# --- UI: Main Panel for Results ---
st.title("💳 Credit Scoring & Risk Analysis Dashboard")
st.markdown("This dashboard uses the production model to predict loan default risk and analyze client features.")
st.markdown("---")

if st.session_state.analysis_generated:
    prob = st.session_state.prob
    shap_explanation_for_plot = st.session_state.shap_explanation_for_plot
    client_data_for_plots = st.session_state.client_data_for_plots
    client_id_for_plots = st.session_state.client_id_for_plots
    prediction = 1 if prob >= BEST_THRESHOLD else 0

    st.header(f"Analysis for: **{client_id_for_plots or 'Custom Client'}**")

    # --- Section 1: Risk Assessment ---
    st.subheader("Risk Assessment")
    status = "High Risk (Default)" if prediction == 1 else "Low Risk (No Default)"
    st.metric("Prediction Status", status)
    st.progress(prob, text=f"Default Probability: {prob:.2%}")
    st.caption(f"A probability above {BEST_THRESHOLD:.0%} is considered high risk.")
    st.markdown("---")

    # --- Section 2: Feature Importance ---
    st.subheader("Feature Contribution to Prediction")
    tab1, tab2 = st.tabs(["Local Importance (This Client)", "Global Importance (All Clients)"])

    with tab1:
        st.markdown("This plot shows how each feature pushed the prediction for **this specific client** from the average score to its final value. Red bars increase risk, blue bars decrease it.")
        try:
            explanation = shap_explanation_for_plot[0]
            shap_values = explanation.values
            base_value = explanation.base_values
            features = np.array(explanation.feature_names)

            feature_impacts = pd.DataFrame({'feature': features, 'shap_value': shap_values})
            feature_impacts['abs_shap'] = feature_impacts['shap_value'].abs()
            feature_impacts = feature_impacts.sort_values(by='abs_shap', ascending=False)

            N = 7
            top_features = feature_impacts.head(N)
            other_features = feature_impacts.iloc[N:]

            if not other_features.empty:
                other_shap_sum = other_features['shap_value'].sum()
                other_row = pd.DataFrame([{'feature': f'{len(other_features)} Other Features', 'shap_value': other_shap_sum}])
                top_features = pd.concat([top_features, other_row], ignore_index=True)

            top_features = top_features.sort_values(by='shap_value', ascending=False)
            y_values = top_features['shap_value'].tolist()
            x_labels = top_features['feature'].tolist()
            final_prediction_from_shap = base_value + np.sum(shap_values)

            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"] * len(y_values) + ["total"],
                x=["Average Prediction"] + x_labels + ["Final Prediction"],
                y=[base_value] + y_values + [final_prediction_from_shap],
                text=[f"{v:.3f}" for v in [base_value] + y_values + [final_prediction_from_shap]],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#d62728"}},
                decreasing={"marker": {"color": "#1f77b4"}},
                totals={"marker": {"color": "#2ca02c"}}
            ))
            fig.update_layout(title="Local Feature Contribution to Prediction", showlegend=False, height=600, yaxis_title="Probability Impact")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plot: {e}")
            st.exception(e)

    with tab2:
        st.markdown("This plot shows the most influential features **across all clients**.")
        if not global_importance_df.empty:
            fig = px.bar(
                global_importance_df.head(20).sort_values('importance', ascending=True),
                x='importance', y='feature', orientation='h',
                title='Top 20 Most Important Features (Global)', text='importance'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Global importance data is not available.")
            
    st.markdown("---")

    # --- Section 3: Client Data Analysis ---
    st.header("Client Data Analysis")
    st.markdown("Compare the current client's features against the distribution of other clients.")
    tab3, tab4 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

    with tab3:
        st.subheader("How does this client compare on a single feature?")
        col1, col2 = st.columns([3, 1])
        with col1:
            feature_to_plot = st.selectbox("Select a feature to analyze", EXPECTED_FEATURES, index=0)
        with col2:
            target_filter = st.selectbox("Filter by Target", ["All", 0, 1], key="univar_target")

        plot_data = analysis_data[analysis_data['TARGET'] == target_filter].copy() if target_filter != "All" else analysis_data.copy()
        
        color_arg_uni = 'TARGET' if 'TARGET' in plot_data.columns and target_filter == "All" else None
        if color_arg_uni:
            plot_data[color_arg_uni] = plot_data[color_arg_uni].astype(str)
            
        fig = px.histogram(
            plot_data, x=feature_to_plot, title=f"Distribution of {feature_to_plot}",
            color=color_arg_uni, color_discrete_map={'0': 'blue', '1': 'red'}
        )
        client_value = client_data_for_plots.get(feature_to_plot)
        fig.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="yellow", annotation_text="Current Client", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

    # --- FINAL CORRECTED BLOCK (v5 - HANDLES ALL FILTER SCENARIOS) ---
    with tab4:
        st.subheader("How does this client compare on two features?")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            x_feature = st.selectbox("Select X-axis feature", EXPECTED_FEATURES, index=1, key="x_feat")
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", EXPECTED_FEATURES, index=2, key="y_feat")
        with col3:
            target_filter_bi = st.selectbox("Filter by Target", ["All", 0, 1], key="bivar_target")

        plot_data_bi = analysis_data.copy()
        if target_filter_bi != "All":
            plot_data_bi = plot_data_bi[plot_data_bi['TARGET'] == int(target_filter_bi)]

        x_is_cat = analysis_data[x_feature].nunique() < 10
        y_is_cat = analysis_data[y_feature].nunique() < 10
        
        fig = None

        # --- Plotting Logic ---

        # Case 1 & 2: Box plots
        if (x_is_cat and not y_is_cat) or (not x_is_cat and y_is_cat):
            color_arg_box = None
            color_map_box = None
            if target_filter_bi == "All":
                plot_data_bi['Status'] = plot_data_bi['TARGET'].map({0: 'No Default', 1: 'Default'})
                color_arg_box = 'Status'
                color_map_box = {'No Default': 'blue', 'Default': 'red'}
            
            fig = px.box(plot_data_bi, x=x_feature, y=y_feature, color=color_arg_box, 
                        title=f"Distribution of {y_feature} by {x_feature}",
                        color_discrete_map=color_map_box)

        # Case 3: Scatter plot
        else:
            # SCATTER: "All" selected -> Build with two colored traces
            if target_filter_bi == "All":
                fig = go.Figure() # Initialize empty figure
                df_0 = plot_data_bi[plot_data_bi['TARGET'] == 0]
                df_1 = plot_data_bi[plot_data_bi['TARGET'] == 1]
                
                fig.add_trace(go.Scatter(x=df_0[x_feature], y=df_0[y_feature], mode='markers', marker=dict(color='blue'), name='No Default (0)'))
                fig.add_trace(go.Scatter(x=df_1[x_feature], y=df_1[y_feature], mode='markers', marker=dict(color='red'), name='Default (1)'))
                
                fig.update_layout(title=f"{x_feature} vs. {y_feature}", xaxis_title=x_feature, yaxis_title=y_feature, legend_title_text='Client Status')
            
            # SCATTER: Filtered for Target 0 -> Make all points BLUE
            elif target_filter_bi == 0:
                fig = px.scatter(plot_data_bi, x=x_feature, y=y_feature, title=f"{x_feature} vs. {y_feature} (No Default)")
                fig.update_traces(marker=dict(color='blue', size=7))
                
            # SCATTER: Filtered for Target 1 -> Make all points RED
            elif target_filter_bi == 1:
                fig = px.scatter(plot_data_bi, x=x_feature, y=y_feature, title=f"{x_feature} vs. {y_feature} (Default)")
                fig.update_traces(marker=dict(color='red', size=7))

        # --- Add Client Marker ---
        client_x_val = client_data_for_plots.get(x_feature)
        client_y_val = client_data_for_plots.get(y_feature)
        
        if client_x_val is not None and client_y_val is not None and fig is not None:
            fig.add_trace(go.Scatter(
                x=[client_x_val], y=[client_y_val], mode='markers', 
                marker=dict(color='yellow', size=15, symbol='star', line=dict(color='black', width=1)), 
                name='Current Client'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    # --- END OF CORRECTED BLOCK ---


else:
    st.info("Select a known client or adjust features in the sidebar and click 'Analyze Client' to begin.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Credit scoring model v1.0. Developed by prêt à dépenser.")