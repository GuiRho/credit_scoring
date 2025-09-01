import streamlit as st
import pandas as pd
import numpy as np # Import numpy
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
# --- NEW: Path to pre-calculated global importance file ---
GLOBAL_IMPORTANCE_PATH = os.path.join(MODEL_PATH, "global_feature_importance.json")
BEST_THRESHOLD = 0.13

# --- Caching Functions for Performance ---
@st.cache_resource
def load_model():
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

# --- NEW: Function to load the pre-calculated global feature importances ---
@st.cache_data
def load_global_importance():
    """Loads the global feature importance data calculated by analysis.py."""
    if os.path.exists(GLOBAL_IMPORTANCE_PATH):
        return pd.read_json(GLOBAL_IMPORTANCE_PATH)
    else:
        # Return an empty dataframe as a fallback
        st.error(f"FATAL: Global importance file not found at '{GLOBAL_IMPORTANCE_PATH}'. Please run analysis.py first.")
        return pd.DataFrame(columns=['feature', 'importance'])

# --- NEW, RECOMMENDED FUNCTION (assuming a tree-based model) ---
@st.cache_resource
def get_shap_explainer(_model, _expected_features):
    # For TreeExplainer, we can pass the model directly.
    # It's much faster and doesn't require a wrapper.
    # The background data is optional but recommended for feature perturbation expectations.
    background_data = load_analysis_data().head(100)
    if 'TARGET' in background_data.columns:
        background_data = background_data.drop(columns=['TARGET'])
        
    # The `model` from mlflow.sklearn is often a pipeline. We need the actual model step.
    # Common step names are 'model', 'classifier', or 'regressor'. Check your training script.
    # If it's not a pipeline, you can just use `_model`.
    if hasattr(_model, 'steps'):
        model_step = _model.steps[-1][1] 
    else:
        model_step = _model
        
    return shap.TreeExplainer(model_step, background_data)

# --- Data and Model Loading ---
model = load_model()
demo_data = load_demo_data()
analysis_data = load_analysis_data()
global_importance_df = load_global_importance() # Load global importance
EXPECTED_FEATURES = list(demo_data.columns)
explainer = get_shap_explainer(model, EXPECTED_FEATURES)

# --- UI: Sidebar for Inputs ---
st.sidebar.title("Client & Feature Controls")
st.sidebar.markdown("---")

# --- MODIFIED: Allow selecting ALL clients from the demo file ---
# The min(5, ...) limit has been removed.
KNOWN_CLIENTS = {
    f"Client_ID_{demo_data.index[i]}": demo_data.iloc[i].to_dict()
    for i in range(len(demo_data))
}
client_id = st.sidebar.selectbox("Select a Known Client (for Demo)", [""] + list(KNOWN_CLIENTS.keys()))

if client_id:
    prefill_data = KNOWN_CLIENTS[client_id]
else:
    prefill_data = demo_data.median().to_dict()

st.sidebar.header("Client Feature Input")
with st.sidebar.expander("Adjust Client Features", expanded=True):
    client_data = {}
    # --- Replacement for the feature input logic in the sidebar ---

for feature in EXPECTED_FEATURES:
    # Use analysis_data for range calculation to cover the full domain
    if feature in analysis_data.columns:
        min_val = analysis_data[feature].min()
        max_val = analysis_data[feature].max()
    else: # Fallback to demo_data if column not in analysis_data
        min_val = demo_data[feature].min()
        max_val = demo_data[feature].max()

    default_val = prefill_data.get(feature, demo_data[feature].median())
    
    # --- IMPROVED LOGIC ---
    
    # Use number_input for floats
    if pd.api.types.is_float_dtype(demo_data[feature]):
        client_data[feature] = st.number_input(
            feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            key=f"num_input_{feature}"
        )
    # Use slider for integers with a small, manageable range
    elif pd.api.types.is_integer_dtype(demo_data[feature]) and (max_val - min_val) < 100:
        client_data[feature] = st.slider(
            feature,
            min_value=int(min_val),
            max_value=int(max_val),
            value=int(default_val),
            step=1,
            key=f"slider_{feature}"
        )
    # Use number_input for integers with a large range (more precise)
    elif pd.api.types.is_integer_dtype(demo_data[feature]):
         client_data[feature] = st.number_input(
            feature,
            min_value=int(min_val),
            max_value=int(max_val),
            value=int(default_val),
            step=1,
            key=f"num_input_{feature}"
        )
    # Use selectbox for categorical/binary features
    else:
        options = sorted(analysis_data[feature].unique())
        try:
            default_index = options.index(default_val)
        except ValueError:
            default_index = 0
        client_data[feature] = st.selectbox(
            feature,
            options,
            index=default_index,
            key=f"select_{feature}"
        )


if st.sidebar.button("Analyze Client", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction and feature contributions..."):
        input_df = pd.DataFrame([client_data], columns=EXPECTED_FEATURES)
        st.session_state.prob = model.predict_proba(input_df)[0, 1]
        # Calculate SHAP explanation object
        shap_explanation = explainer(input_df)
        
        # --- NEW: Create a new explanation object with rounded values for plotting ---
        # We target the positive class (index 1) for default prediction.
        rounded_values = np.round(shap_explanation.values[:, :, 1], 2)
        st.session_state.shap_explanation_rounded = shap.Explanation(
            values=rounded_values,
            base_values=shap_explanation.base_values[:, 1],
            data=shap_explanation.data,
            feature_names=EXPECTED_FEATURES
        )
        
        st.session_state.client_data_for_plots = client_data
        st.session_state.client_id_for_plots = client_id
        st.session_state.analysis_generated = True

# --- UI: Main Panel for Results ---
st.title("💳 Credit Scoring & Risk Analysis Dashboard")
st.markdown("This dashboard uses the production model to predict loan default risk and analyze client features.")
st.markdown("---")

if st.session_state.get("analysis_generated", False):
    prob = st.session_state.prob
    # Use the rounded explanation for plotting
    shap_explanation_rounded = st.session_state.shap_explanation_rounded
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

    # --- Section 2: Feature Importance (Local vs Global) ---
    st.subheader("Feature Contribution to Prediction")
    
    # --- MODIFIED: Use tabs for local and global importance ---
    tab1, tab2 = st.tabs(["Local Importance (This Client)", "Global Importance (All Clients)"])

    with tab1:
        st.markdown("This plot shows how each feature pushed the prediction for **this specific client** from the average score to its final value. Red bars increase risk, blue bars decrease it.")
        fig, ax = plt.subplots()
        # Use the pre-rounded explanation object for the plot
        shap.plots.waterfall(shap_explanation_rounded[0], max_display=15, show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)

    with tab2:
        st.markdown("This plot shows the most influential features **across all clients**. It is based on the average impact (mean absolute SHAP value) each feature has on the prediction.")
        if not global_importance_df.empty:
            # Create a bar chart with Plotly
            fig = px.bar(
                global_importance_df.head(20).sort_values('importance', ascending=True),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 20 Most Important Features (Global)',
                text='importance' # Show values on bars
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
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

        # Filter data based on selection
        if target_filter != "All":
            plot_data = analysis_data[analysis_data['TARGET'] == target_filter]
        else:
            plot_data = analysis_data

        fig = px.histogram(
            plot_data, 
            x=feature_to_plot, 
            title=f"Distribution of {feature_to_plot}",
            color='TARGET' if 'TARGET' in plot_data.columns and target_filter == "All" else None,
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        
        client_value = client_data_for_plots.get(feature_to_plot)
        fig.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="green", annotation_text="Current Client", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("How does this client compare on two features?")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            x_feature = st.selectbox("Select X-axis feature", EXPECTED_FEATURES, index=1, key="x_feat")
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", EXPECTED_FEATURES, index=2, key="y_feat")
        with col3:
            target_filter_bi = st.selectbox("Filter by Target", ["All", 0, 1], key="bivar_target")

        # Filter data
        if target_filter_bi != "All":
            plot_data_bi = analysis_data[analysis_data['TARGET'] == target_filter_bi]
        else:
            plot_data_bi = analysis_data

        # Determine plot type
        x_is_cat = analysis_data[x_feature].nunique() < 20
        y_is_cat = analysis_data[y_feature].nunique() < 20
        
        color_arg = 'TARGET' if 'TARGET' in plot_data_bi.columns and target_filter_bi == "All" else None

        if x_is_cat and not y_is_cat:
            fig = px.box(plot_data_bi, x=x_feature, y=y_feature, color=color_arg, title=f"{y_feature} by {x_feature}", color_discrete_map={0: 'blue', 1: 'red'})
        elif not x_is_cat and y_is_cat:
            fig = px.box(plot_data_bi, x=y_feature, y=x_feature, color=color_arg, title=f"{x_feature} by {y_feature}", color_discrete_map={0: 'blue', 1: 'red'})
        else: # Both numeric or both categorical
            fig = px.scatter(plot_data_bi, x=x_feature, y=y_feature, color=color_arg, title=f"{x_feature} vs. {y_feature}", opacity=0.6, color_discrete_map={0: 'blue', 1: 'red'})

        client_x, client_y = client_data_for_plots.get(x_feature), client_data_for_plots.get(y_feature)
        if not x_is_cat and not y_is_cat:
             fig.add_trace(go.Scatter(x=[client_x], y=[client_y], mode='markers', marker=dict(color='green', size=15, symbol='star'), name='Current Client'))
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select a known client or adjust features in the sidebar and click 'Analyze Client' to begin.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("This app runs in offline mode, using a local copy of the production model for analysis.")