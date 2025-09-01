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
    """
    Creates a SHAP TreeExplainer. This is much faster and more reliable
    for tree-based models like LightGBM, XGBoost, etc.
    """
    # MLflow models are often pipelines. We need to extract the final model step.
    # Common step names are 'model', 'classifier', or the model's class name.
    # Check your model training script to confirm the step name.
    if hasattr(_model, 'steps'):
        model_step = _model.steps[-1][1] 
    else: # If the model is not a pipeline
        model_step = _model
        
    # TreeExplainer is initialized with the model itself.
    # It does not need a prediction wrapper function.
    return shap.TreeExplainer(model_step)

# --- Data and Model Loading ---
model = load_model()
analysis_data = load_analysis_data()
global_importance_df = load_global_importance() # Load global importance
EXPECTED_FEATURES = list(analysis_data.drop(columns=['TARGET']).columns)
explainer = get_shap_explainer(model, EXPECTED_FEATURES)

# --- UI: Sidebar for Inputs ---
st.sidebar.title("Client & Feature Controls")
st.sidebar.markdown("---")

# --- MODIFIED: Allow selecting ALL clients from the analysis file ---
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
# --- Replacement for the feature input logic in the sidebar ---

with st.sidebar.expander("Adjust Client Features", expanded=True):
    client_data = {}
    for feature in EXPECTED_FEATURES:
        
        # --- IMPROVED & ROBUST LOGIC ---
        
        # Step 1: Get feature properties consistently from the main analysis_data
        if feature in analysis_data.columns:
            series = analysis_data[feature]
            min_val = series.min()
            max_val = series.max()
            dtype = series.dtype
        else: # Fallback to demo_data if feature is missing from analysis_data
            series = analysis_data[feature] # This should not happen anymore
            min_val = series.min()
            max_val = series.max()
            dtype = series.dtype

        default_val = prefill_data.get(feature, series.median())

        # Step 2: FIX - Handle constant features to prevent the slider error
        if min_val == max_val:
            # Display the value as non-editable text and add it to the client data
            st.text_input(f"{feature} (Constant)", value=default_val, disabled=True)
            client_data[feature] = default_val 
            continue # Skip to the next feature in the loop

        # Step 3: Choose the best UI element based on data type and range
        
        # Use slider for all numerical features (floats and integers)
        if pd.api.types.is_numeric_dtype(dtype):
            client_data[feature] = st.slider(
                feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                key=f"slider_{feature}"
            )
        # Use radio buttons for categorical features
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

    # --- FINAL Replacement for the "tab1" (Local Importance) plotting logic ---

    with tab1:
        st.markdown("This plot shows how each feature pushed the prediction for **this specific client** from the average score to its final value. Red bars increase risk, blue bars decrease it.")

        try:
            explanation = shap_explanation_rounded[0]
            
            # --- IMPROVED DATA PREPARATION ---
            
            # 1. Extract data and create a DataFrame for easy manipulation
            shap_values = explanation.values
            base_value = explanation.base_values
            features = np.array(explanation.feature_names)
            
            feature_impacts = pd.DataFrame(
                {'feature': features, 'shap_value': shap_values}
            )
            feature_impacts['abs_shap'] = feature_impacts['shap_value'].abs()
            feature_impacts = feature_impacts.sort_values(by='abs_shap', ascending=False)

            # 2. Separate the top N features from the rest
            N = 7
            top_features = feature_impacts.head(N)
            other_features = feature_impacts.iloc[N:]

            # 3. FIX: Calculate the sum of the remaining features' impact
            if not other_features.empty:
                other_shap_sum = other_features['shap_value'].sum()
                # Append the 'Other Features' contribution to the top features DataFrame
                other_row = pd.DataFrame([{
                    'feature': f'{len(other_features)} Other Features',
                    'shap_value': other_shap_sum
                }])
                top_features = pd.concat([top_features, other_row], ignore_index=True)

            # Sort the bars for a cleaner visual flow (optional, but good practice)
            top_features = top_features.sort_values(by='shap_value', ascending=False)

            # 4. Prepare final data lists for Plotly
            y_values = top_features['shap_value'].tolist()
            x_labels = top_features['feature'].tolist()
            final_prediction = st.session_state.prob # Use the model's actual prediction

            # 5. Create the Plotly figure with the corrected data
            fig = go.Figure(go.Waterfall(
                name="Prediction",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(y_values) + ["total"],
                x=["Average Prediction"] + x_labels + ["Final Prediction"],
                y=[base_value] + y_values + [final_prediction],
                text=[f"{v:.3f}" for v in [base_value] + y_values + [final_prediction]],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#d62728"}}, # Red for increasing risk
                decreasing={"marker": {"color": "#1f77b4"}}, # Blue for decreasing risk
                totals={"marker": {"color": "#2ca02c"}}     # Green for totals
            ))

            fig.update_layout(
                title="Local Feature Contribution to Prediction",
                showlegend=False,
                height=600,
                yaxis_title="Probability Impact"
            )
            # Rotate x-axis labels to prevent them from overlapping
            fig.update_xaxes(tickangle=-45)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plot: {e}")
            st.exception(e) # This will print the full traceback for debugging

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