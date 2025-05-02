import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Removed as not used
# import seaborn as sns # Removed as not used
import io
# from matplotlib.backends.backend_pdf import PdfPages # Removed as PDF export is removed
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots # Removed as not used
import traceback # Keep for debugging

# Set page config
st.set_page_config(page_title="Bar Exam Model Bias Analysis", layout="wide")
st.title("Bar Exam Model Bias Analysis")
st.markdown("Analyze option selection bias for specific model configurations.")

# --- Helper Functions ---

def clean_label(text):
    """Make column names and labels more readable."""
    if pd.isna(text) or text is None or text == "None":
        return ""
    clean_text = str(text).replace('_', ' ')
    clean_text = ' '.join(word.capitalize() for word in clean_text.split())
    return clean_text

# --- Data Loading ---
# Sidebar configuration
st.sidebar.title("Configuration")

# Hardcoded file path (Consider making this an input or upload)
file_path = "ALL_llama_new_question_level_analysis_results.csv"
st.sidebar.caption(f"Loading data from: {file_path}")

@st.cache_data
def load_data(path):
    """Loads and performs initial processing on the CSV data."""
    try:
        df = pd.read_csv(path, low_memory=False)
        st.sidebar.info(f"Loaded {len(df)} raw records.")
        # Filter out rows where file_exists is False if column exists
        if 'file_exists' in df.columns:
            original_count = len(df)
            df = df[df['file_exists'] == True].copy() # Use .copy() to avoid SettingWithCopyWarning
            st.sidebar.info(f"Removed {original_count - len(df)} records where 'file_exists' was False.")

        # Basic type conversions needed for filtering/analysis
        if 'training_status' in df.columns:
            df['training_status'] = df['training_status'].astype(str)
        if 'model_name' in df.columns:
            df['model_name'] = df['model_name'].astype(str)
        # Convert potential label columns to string upfront
        for col in ['predicted_label', 'ground_truth_label']:
             if col in df.columns:
                 df[col] = df[col].astype(str)

        # Deduplication
        original_count_dedup = len(df)
        df.drop_duplicates(inplace=True, keep='first')
        num_duplicates = original_count_dedup - len(df)
        if num_duplicates > 0:
            st.sidebar.info(f"Removed {num_duplicates} duplicate rows.")
        st.sidebar.info(f"Using {len(df)} unique records.")

        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at '{path}'. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.code(traceback.format_exc())
        return None

# Load data
df_loaded = load_data(file_path)

# --- Main App Logic ---
if df_loaded is not None and not df_loaded.empty:

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Options")
    st.sidebar.markdown("Select the specific configuration to analyze.")

    # REVISION: Single selection for Model Name
    model_options = sorted(df_loaded['model_name'].unique())
    selected_model = st.sidebar.selectbox(
        "Select Model Name",
        options=model_options,
        index=0,
        key="select_model",
        help="Choose the model you want to analyze."
    )

    # REVISION: Single selection for Training Status
    training_options = sorted(df_loaded['training_status'].unique())
    selected_training_status = st.sidebar.radio(
        "Select Training Status",
        options=training_options,
        index=0,
        key="select_training",
        horizontal=True,
        help="Analyze either the 'untrained' or 'trained' version."
    )

    # --- Apply Core Filters (Model & Training Status) ---
    df_core_filtered = df_loaded[
        (df_loaded['model_name'] == selected_model) &
        (df_loaded['training_status'] == selected_training_status)
    ].copy() # Use copy to avoid warnings

    st.sidebar.divider()
    st.sidebar.markdown("**Additional Filters**")

    # --- Additional Parameter Filters (Multiselect) ---
    active_filters = {} # Store other active filters

    # Generation Parameters
    st.sidebar.markdown("**Generation Parameters**")
    gen_params_to_filter = ['generation', 'response_format', 'response_type', 'prompt_type', 'explanation_type']
    for param in gen_params_to_filter:
        if param in df_core_filtered.columns:
            options = sorted(df_core_filtered[param].unique())
            if len(options) > 1: # Only show filter if there's more than one option
                # Use session state for remembering multiselect defaults
                filter_key = f"filter_{param}"
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = options # Default to all selected
                selected_vals = st.sidebar.multiselect(f"{clean_label(param)}", options=options, key=filter_key)
                active_filters[param] = selected_vals

    # Dataset Filters
    st.sidebar.markdown("**Dataset Parameters**")
    dataset_params_to_filter = ['dataset', 'eval_dataset'] # Removed 'quantisation'
    for param in dataset_params_to_filter:
         if param in df_core_filtered.columns:
            options = sorted(df_core_filtered[param].unique())
            if len(options) > 1:
                filter_key = f"filter_{param}"
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = options
                selected_vals = st.sidebar.multiselect(f"{clean_label(param)}", options=options, key=filter_key)
                active_filters[param] = selected_vals

    # Apply additional filters
    df_final_filtered = df_core_filtered.copy()
    filter_summary = [f"Model: **{selected_model}**", f"Training: **{selected_training_status}**"]
    for col, selected_values in active_filters.items():
        if selected_values and len(selected_values) < len(df_final_filtered[col].unique()): # Apply filter only if not all values are selected
            df_final_filtered = df_final_filtered[df_final_filtered[col].isin(selected_values)]
            filter_summary.append(f"{clean_label(col)}: {', '.join(map(str, selected_values))}")

    st.sidebar.divider()
    st.sidebar.markdown(f"**Analysis Scope:** Analyzing **{len(df_final_filtered)}** records for:")
    for item in filter_summary:
        st.sidebar.markdown(f"- {item}")

    # --- Grouping Parameter Selection ---
    st.sidebar.subheader("Group By Parameter")
    st.sidebar.markdown("Select ONE parameter to group the bias analysis by.")
    # Potential grouping parameters (exclude model and training status as they are fixed)
    potential_group_params = [
        col for col in ['generation', 'response_format', 'response_type', 'prompt_type', 'explanation_type', 'dataset', 'eval_dataset']
        if col in df_final_filtered.columns and df_final_filtered[col].nunique() > 1
    ]
    # Add 'None' option to see overall bias without grouping
    group_by_param = st.sidebar.selectbox(
        "Group Analysis By",
        options=["None"] + potential_group_params,
        index=0, # Default to 'None'
        format_func=lambda x: "Overall (No Grouping)" if x == "None" else clean_label(x),
        help="Optional: Group the bias results by the selected parameter."
    )

    # --- Plot Color Palette ---
    st.sidebar.subheader("Plot Settings")
    color_palette = st.sidebar.selectbox(
        "Color Palette",
        options=['Plotly', 'Set1', 'Set2', 'Set3','Alphabet','D3', 'Bold', 'Prism', 'Dark24', 'Light24'],
        index=0, help="Select color scheme for plots."
    )
    try:
        colors = getattr(px.colors.qualitative, color_palette)
    except AttributeError:
        st.sidebar.warning(f"Palette {color_palette} not found, using Plotly.")
        colors = px.colors.qualitative.Plotly

    # --- Main Analysis Area ---
    if not df_final_filtered.empty:
        # Display context clearly
        st.markdown("### Current Analysis Configuration")
        st.markdown(f"**Model:** `{selected_model}` | **Training Status:** `{selected_training_status}`")
        if group_by_param != "None":
             st.markdown(f"**Grouped By:** `{clean_label(group_by_param)}`")
        else:
             st.markdown("**Grouped By:** `Overall (No Grouping)`")
        # Display other active filters if any
        other_filters_applied = [f"{clean_label(k)}: {', '.join(map(str, v))}" for k, v in active_filters.items() if v and len(v) < len(df_core_filtered[k].unique())]
        if other_filters_applied:
             st.markdown(f"**Additional Filters:** `{'; '.join(other_filters_applied)}`")
        st.divider()


        # --- Tabbed Analysis ---
        # REVISION: Removed "Model Struggle Points" and "Parameter Influence" tabs
        tabs = st.tabs(["📊 Option Selection Bias", "❓ Label Confusion Analysis"])

        # ================= Tab 1: Option Selection Bias =================
        with tabs[0]:
            st.header("Option Selection Bias")
            st.markdown("Compares the distribution of the model's predicted answers (A, B, C, D) against the distribution of the correct answers.")

            # --- Overall Bias Calculation (Always calculated for reference) ---
            pred_option_counts_overall = df_final_filtered['predicted_label'].value_counts(normalize=True) * 100
            gt_option_counts_overall = df_final_filtered['ground_truth_label'].value_counts(normalize=True) * 100
            all_options = sorted(list(set(pred_option_counts_overall.index) | set(gt_option_counts_overall.index)))
            option_bias_data_overall = []
            for option in all_options:
                pred_pct = pred_option_counts_overall.get(option, 0)
                gt_pct = gt_option_counts_overall.get(option, 0)
                option_bias_data_overall.append({
                    'Option': option, 'Predicted (%)': pred_pct, 'Ground Truth (%)': gt_pct, 'Difference (%)': pred_pct - gt_pct
                })
            option_bias_df_overall = pd.DataFrame(option_bias_data_overall).sort_values('Option')
            rmse_overall = np.sqrt(np.mean(option_bias_df_overall['Difference (%)']**2))

            # --- Grouped Analysis (if group_by_param is selected) ---
            if group_by_param != "None":
                st.subheader(f"Bias Analysis Grouped by {clean_label(group_by_param)}")

                # Calculate metrics per group
                grouped_data = df_final_filtered.groupby(group_by_param)
                bias_metrics_grouped = []
                all_group_option_data = []

                for name, group in grouped_data:
                    pred_counts_group = group['predicted_label'].value_counts(normalize=True) * 100
                    gt_counts_group = group['ground_truth_label'].value_counts(normalize=True) * 100
                    group_bias = {}
                    group_option_data = []
                    sum_sq_diff = 0
                    for option in all_options:
                        pred_pct = pred_counts_group.get(option, 0)
                        gt_pct = gt_counts_group.get(option, 0)
                        diff = pred_pct - gt_pct
                        sum_sq_diff += diff**2
                        group_bias[f'Bias_{option}'] = diff
                        # Data for grouped bar chart
                        group_option_data.append({'Group': name, 'Option': option, 'Percentage': pred_pct, 'Type': 'Predicted'})
                        group_option_data.append({'Group': name, 'Option': option, 'Percentage': gt_pct, 'Type': 'Ground Truth'})

                    rmse_group = np.sqrt(sum_sq_diff / len(all_options))
                    bias_metrics_grouped.append({ 'Group': name, 'RMSE_Bias': rmse_group, **group_bias })
                    all_group_option_data.extend(group_option_data)

                bias_df_grouped = pd.DataFrame(bias_metrics_grouped)
                all_group_option_df = pd.DataFrame(all_group_option_data)

                # Plot Grouped Option Distribution Comparison
                fig_options_grouped = px.bar(
                    all_group_option_df, x='Group', y='Percentage', color='Type', barmode='group',
                    facet_col='Option', category_orders={"Option": all_options},
                    title=f"Predicted vs Ground Truth Option Distribution by {clean_label(group_by_param)}",
                    labels={'Group': clean_label(group_by_param), 'Percentage': 'Percentage (%)'},
                    color_discrete_map={'Predicted': colors[0], 'Ground Truth': colors[1]}
                )
                fig_options_grouped.update_xaxes(tickangle=45)
                st.plotly_chart(fig_options_grouped, use_container_width=True)

                # Plot Grouped RMSE Bias
                fig_bias_grouped = px.bar(
                    bias_df_grouped.sort_values('RMSE_Bias', ascending=False),
                    x='Group', y='RMSE_Bias', title=f"Overall Option Selection Bias (RMSE) by {clean_label(group_by_param)}",
                    labels={'Group': clean_label(group_by_param), 'RMSE_Bias': 'RMSE Bias (%)'},
                    color='RMSE_Bias', color_continuous_scale='Reds'
                )
                fig_bias_grouped.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bias_grouped, use_container_width=True)

                # Display Grouped Bias Table
                st.dataframe(bias_df_grouped.style.format({'RMSE_Bias': '{:.2f}', **{f'Bias_{opt}': '{:.2f}' for opt in all_options}}), use_container_width=True)

            # --- Always show Overall Bias ---
            st.subheader("Overall Option Selection Bias (Across Filtered Data)")
            col1, col2 = st.columns([2,1])
            with col1:
                # Plot Overall Option Distribution Comparison
                combined_overall_counts = pd.concat([
                    option_bias_df_overall[['Option', 'Predicted (%)']].rename(columns={'Predicted (%)':'Percentage'}).assign(Type='Predicted'),
                    option_bias_df_overall[['Option', 'Ground Truth (%)']].rename(columns={'Ground Truth (%)':'Percentage'}).assign(Type='Ground Truth')
                ])
                fig_options_overall = px.bar(
                    combined_overall_counts, x='Option', y='Percentage', color='Type', barmode='group',
                    title="Overall Distribution of Options (A, B, C, D)", labels={'Percentage': 'Percentage (%)'},
                    color_discrete_map={'Predicted': colors[0], 'Ground Truth': colors[1]}
                )
                st.plotly_chart(fig_options_overall, use_container_width=True)
            with col2:
                 st.metric("Overall Option Selection Bias (RMSE)", f"{rmse_overall:.2f}%",
                           help="Root Mean Square Error between overall predicted and ground truth option distributions. Lower values indicate less overall bias.")
                 st.dataframe(option_bias_df_overall.style.format({'Predicted (%)': '{:.2f}', 'Ground Truth (%)': '{:.2f}', 'Difference (%)': '{:.2f}'}), use_container_width=True)

            st.divider()
            # --- Option Selection Bias by True Label ---
            st.subheader("Option Selection Bias by Ground Truth Label")
            st.markdown("Analyzes if the model prefers predicting certain options when the *actual* correct answer is A, B, C, or D.")

            pred_label_option_pivot = pd.crosstab(df_final_filtered['ground_truth_label'], df_final_filtered['predicted_label'], normalize='index') * 100
            # Ensure all options A,B,C,D are present as columns/index, fill missing with 0
            all_options_labels = ['A', 'B', 'C', 'D'] # Assuming these are the only options
            pred_label_option_pivot = pred_label_option_pivot.reindex(index=all_options_labels, columns=all_options_labels, fill_value=0)

            # Calculate bias (difference from perfect prediction where Predicted = Ground Truth)
            bias_label_option_pivot = pred_label_option_pivot.copy()
            for idx in bias_label_option_pivot.index:
                bias_label_option_pivot.loc[idx, idx] -= 100 # Subtract 100 from the diagonal (perfect prediction)

            bias_tabs = st.tabs(["Predicted Distribution Heatmap", "Bias Heatmap", "Bias by Label (RMSE)"])
            with bias_tabs[0]:
                fig_pred_heatmap = px.imshow(
                    pred_label_option_pivot, labels=dict(x="Predicted Option", y="Ground Truth Label", color="Prediction %"),
                    x=all_options_labels, y=all_options_labels, color_continuous_scale="Blues",
                    title="Heatmap: What the Model Predicts Given the True Answer"
                )
                fig_pred_heatmap.update_layout(coloraxis_colorbar=dict(title="Prediction %"))
                st.plotly_chart(fig_pred_heatmap, use_container_width=True)
            with bias_tabs[1]:
                fig_bias_heatmap = px.imshow(
                    bias_label_option_pivot, labels=dict(x="Predicted Option", y="Ground Truth Label", color="Bias %"),
                    x=all_options_labels, y=all_options_labels, color_continuous_scale="RdBu", color_continuous_midpoint=0,
                    title="Heatmap: Option Prediction Bias Given the True Answer"
                )
                fig_bias_heatmap.update_layout(coloraxis_colorbar=dict(title="Bias %"))
                st.plotly_chart(fig_bias_heatmap, use_container_width=True)
            with bias_tabs[2]:
                rmse_by_label = bias_label_option_pivot.apply(lambda row: np.sqrt(np.mean(row**2)), axis=1).reset_index()
                rmse_by_label.columns = ['Ground Truth Label', 'RMSE_Bias']
                fig_rmse_bar = px.bar(
                    rmse_by_label.sort_values('RMSE_Bias', ascending=False), x='Ground Truth Label', y='RMSE_Bias',
                    title="Overall Bias (RMSE) for Each Ground Truth Label", labels={'RMSE_Bias': 'RMSE Bias (%)'},
                    color='RMSE_Bias', color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_rmse_bar, use_container_width=True)


        # ================= Tab 2: Label Confusion Analysis =================
        with tabs[1]:
            st.header("Label Confusion Analysis")
            st.markdown("Shows how often the model predicts one label when the ground truth is another.")

            # --- Overall Confusion Matrix ---
            st.subheader("Overall Label Confusion Matrix (Across Filtered Data)")
            try:
                label_confusion = pd.crosstab(
                    df_final_filtered['ground_truth_label'],
                    df_final_filtered['predicted_label'],
                    normalize='index' # Normalize by row (ground truth)
                ) * 100
                # Ensure all A,B,C,D are present
                label_confusion = label_confusion.reindex(index=all_options_labels, columns=all_options_labels, fill_value=0)

                fig_label_confusion = px.imshow(
                    label_confusion, labels=dict(x="Predicted Label", y="Ground Truth Label", color="Prediction %"),
                    x=all_options_labels, y=all_options_labels, color_continuous_scale="Blues",
                    title="Confusion Matrix: % of Times Each Label Was Predicted Given the Ground Truth"
                )
                fig_label_confusion.update_layout(coloraxis_colorbar=dict(title="%"))
                st.plotly_chart(fig_label_confusion, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate overall confusion matrix: {e}")


            # --- Grouped Confusion Analysis ---
            if group_by_param != "None":
                st.subheader(f"Label Accuracy by {clean_label(group_by_param)}")
                st.markdown(f"Shows the percentage of correct predictions for each ground truth label, broken down by **{clean_label(group_by_param)}**.")

                try:
                    # Calculate label match accuracy per group and per ground truth label
                    df_final_filtered['label_match'] = (df_final_filtered['ground_truth_label'] == df_final_filtered['predicted_label']).astype(int)
                    label_acc_by_group = df_final_filtered.groupby([group_by_param, 'ground_truth_label'])['label_match'].mean().reset_index()
                    label_acc_by_group['Accuracy (%)'] = label_acc_by_group['label_match'] * 100

                    fig_label_acc_grouped = px.bar(
                        label_acc_by_group, x='ground_truth_label', y='Accuracy (%)', color=group_by_param,
                        title=f"Label Prediction Accuracy by {clean_label(group_by_param)}",
                        labels={'ground_truth_label': 'Ground Truth Label'}, barmode='group',
                        category_orders={"ground_truth_label": all_options_labels},
                        color_discrete_sequence=colors
                    )
                    st.plotly_chart(fig_label_acc_grouped, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate grouped label accuracy plot: {e}")

    # --- Display Raw Data ---
    st.divider()
    with st.expander("View Filtered Data Sample"):
        st.markdown(f"Showing the first 100 rows of the **{len(df_final_filtered)}** currently filtered data points.")
        st.dataframe(df_final_filtered.head(100))

else:
    st.error("Failed to load or process data. Cannot display analysis.")
