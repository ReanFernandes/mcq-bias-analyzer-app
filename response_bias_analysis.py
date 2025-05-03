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
from collections.abc import Iterable # Import Iterable
import plotly.io as pio

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

def sort_data_by_column(df, column, sort_order="Ascending"):
    """Sort dataframe by a column in specified order."""
    if sort_order == "No Sorting" or column is None or column == "None" or column not in df.columns:
        return df

    df_copy = df.copy() # Work on a copy

    # Attempt numeric sort first
    numeric_col = pd.to_numeric(df_copy[column], errors='coerce')
    if not numeric_col.isna().all(): # If at least some values are numeric
        df_copy['_sort_val_'] = numeric_col
        return df_copy.sort_values(by='_sort_val_', ascending=(sort_order == "Ascending"), na_position='last').drop('_sort_val_', axis=1)
    else:
        # Fallback to string sorting, handle potential mixed types
        try:
            # Ensure all values are strings before sorting
            return df_copy.sort_values(by=column, key=lambda col: col.astype(str), ascending=(sort_order == "Ascending"), na_position='last')
        except TypeError: # Final fallback if conversion fails
             return df_copy.sort_values(by=column, ascending=(sort_order == "Ascending"), na_position='last')


# --- Plotting & Analysis Functions ---

def power_law(x, a, b, c):
    """Power law function for fitting learning curves: y = a - b * x^(-c)."""
    x_numeric = pd.to_numeric(x, errors='coerce')
    # Replace NaNs or non-positives with a small epsilon
    x_safe = np.where(np.isnan(x_numeric) | (x_numeric <= 0), 1e-10, x_numeric)
    # Avoid potential overflow or invalid values in power calculation
    with np.errstate(over='ignore', invalid='ignore'):
        power_term = np.power(x_safe, -c)
        # Replace inf or nan results from power term with a large number or zero
        power_term = np.nan_to_num(power_term, nan=0.0, posinf=1e10) # Use 1e10 as large number
    return a - b * power_term

# MODIFICATION: Accept group_col as parameter
def analyze_learning_curve(data, group_col='model_name'):
    """
    Analyze learning curves for each group in the dataset.
    Assumes 'num_training_samples', 'label_accuracy', and the specified group_col columns.
    """
    required_cols = [group_col, 'num_training_samples', 'label_accuracy']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns for learning curve analysis: Need {', '.join(required_cols)}")
        return pd.DataFrame(), {}, {}

    data = data.copy()
    data['num_training_samples'] = pd.to_numeric(data['num_training_samples'], errors='coerce')
    data['label_accuracy'] = pd.to_numeric(data['label_accuracy'], errors='coerce')
    data = data.dropna(subset=['num_training_samples', 'label_accuracy', group_col])

    if data.empty:
        st.warning("No valid data for learning curve analysis after cleaning.")
        return pd.DataFrame(), {}, {}

    # Group by the specified group_col and sample size
    grouped = data.groupby([group_col, 'num_training_samples'])['label_accuracy'].agg(
        median='median', mean='mean', std='std',
        q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), count='count'
    ).reset_index()

    groups = data[group_col].unique()
    fit_params = {}
    r_squared = {}

    for group in groups:
        model_data = grouped[grouped[group_col] == group].sort_values('num_training_samples')
        if len(model_data) < 3: continue

        x = model_data['num_training_samples'].values
        y = model_data['median'].values
        valid_mask = (x > 0) & (~np.isnan(y)) # Ensure x is positive and y is not NaN
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < 3: continue

        try:
            # Define bounds more carefully based on data range
            y_min_data, y_max_data = y_valid.min(), y_valid.max()
            a_upper_bound = max(1.0, y_max_data * 1.5) # Allow asymptote above max observed
            a_lower_bound = min(0.0, y_min_data * 0.5) # Allow asymptote below min observed
            p0 = [y_max_data, 0.1, 0.1] # Adjust initial guess
            bounds = ([a_lower_bound, 0, 1e-9], [a_upper_bound, np.inf, np.inf]) # Bounds for a, b, c (ensure c > 0)

            popt, pcov = curve_fit(power_law, x_valid, y_valid, p0=p0, bounds=bounds, maxfev=5000, method='trf')

            fit_params[group] = popt
            y_pred = power_law(x_valid, *popt)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared[group] = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 1.0 # Avoid division by zero

        except (RuntimeError, ValueError, Exception) as e: # Catch specific and general errors
            # st.warning(f"Could not fit curve for {group}: {e}") # Reduce verbosity
            fit_params[group] = None # Explicitly mark as failed
            r_squared[group] = None

    return grouped, fit_params, r_squared

def plot_learning_curves_plotly(grouped, fit_params, r_squared, human_baseline=0.675, random_chance=0.25, use_log_x=True, show_error_bars=True, colors=px.colors.qualitative.Plotly, group_col='model_name'):
    """Create a Plotly figure of learning curves."""
    fig = go.Figure()
    # Ensure group_col exists before proceeding
    if group_col not in grouped.columns:
         st.error(f"Grouping column '{group_col}' not found in the grouped learning curve data.")
         return fig # Return empty figure

    models = grouped[group_col].unique()
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    all_x = grouped['num_training_samples'].unique()
    min_x_overall = all_x[all_x > 0].min() if any(all_x > 0) else 1
    max_x_overall = all_x.max() if len(all_x) > 0 else 100
    min_y_overall = grouped['median'].min() if not grouped.empty else 0
    max_y_overall = grouped['median'].max() if not grouped.empty else 1

    for model in models:
        model_data = grouped[grouped[group_col] == model].sort_values('num_training_samples')
        if len(model_data) < 1: continue

        error_y_config = None
        if show_error_bars:
             error_y_config=dict(type='data', symmetric=False, array=model_data['q75'] - model_data['median'], arrayminus=model_data['median'] - model_data['q25'], thickness=1, width=3)

        fig.add_trace(go.Scatter(
            x=model_data['num_training_samples'], y=model_data['median'],
            error_y=error_y_config, mode='markers+lines' if len(model_data) > 1 else 'markers',
            name=f"{str(model)} (Median)", marker=dict(color=model_colors.get(model, 'grey'), size=8), # Use .get for safety
            line=dict(color=model_colors.get(model, 'grey'), width=1.5), legendgroup=str(model)
        ))

        if model in fit_params and fit_params[model] is not None: # Check for successful fit
            params = fit_params[model]
            r2 = r_squared.get(model)
            r2_text = f", R²={r2:.3f}" if r2 is not None else ""

            min_x_fit = model_data['num_training_samples'][model_data['num_training_samples'] > 0].min() if any(model_data['num_training_samples'] > 0) else min_x_overall
            max_x_fit = model_data['num_training_samples'].max() * 1.2
            min_x_fit_safe = max(min_x_fit, 1e-9) # Ensure positive for logspace

            if use_log_x:
                 x_smooth = np.logspace(np.log10(min_x_fit_safe), np.log10(max_x_fit), 100)
            else:
                x_smooth = np.linspace(min_x_fit, max_x_fit, 100)

            x_smooth_safe = x_smooth[x_smooth > 0]
            if len(x_smooth_safe) > 0:
                y_smooth = power_law(x_smooth_safe, *params)
                fig.add_trace(go.Scatter(
                    x=x_smooth_safe, y=y_smooth, mode='lines',
                    name=f"{str(model)} (Fit{r2_text})",
                    line=dict(color=model_colors.get(model, 'grey'), dash='dash', width=2),
                    legendgroup=str(model)
                ))

    fig.add_hline(y=human_baseline, line=dict(color="purple", width=2, dash="dash"), annotation_text="Human Baseline", annotation_position="top right")
    fig.add_hline(y=random_chance, line=dict(color="gray", width=2, dash="dash"), annotation_text="Random Chance", annotation_position="bottom right")

    xaxis_config = dict(title="Number of Training Samples", showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
    if use_log_x:
        xaxis_config['type'] = "log"; xaxis_config['exponentformat'] = "power"
        min_x_range = max(min_x_overall, 1e-9)
        xaxis_config['range'] = [np.log10(min_x_range), np.log10(max_x_overall * 1.5)]
    else:
        xaxis_config['type'] = "linear"; xaxis_config['range'] = [0, max_x_overall * 1.1]

    fig.update_layout(
        title="Learning Curve: Accuracy vs. Number of Training Samples", xaxis=xaxis_config,
        yaxis=dict(title="Median Label Accuracy", range=[0, 1], showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)'),
        legend=dict(title=clean_label(group_col), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1),
        hovermode="closest", template="plotly_white", margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# MODIFICATION: Accept group_col as parameter
def calculate_efficiency_metrics(grouped, fit_params, group_col='model_name'):
    """Calculate various efficiency metrics."""
    metrics = {}
    # Ensure group_col exists
    if group_col not in grouped.columns:
         st.error(f"Grouping column '{group_col}' not found for efficiency metrics calculation.")
         return metrics

    groups = grouped[group_col].unique()

    for group in groups:
        model_data = grouped[grouped[group_col] == group].sort_values('num_training_samples')
        if len(model_data) < 2: continue

        min_positive_samples_data = model_data[model_data['num_training_samples'] > 0]
        if min_positive_samples_data.empty: continue # Skip if no positive samples
        baseline_row = min_positive_samples_data.iloc[0]
        baseline_x = baseline_row['num_training_samples']
        baseline_y = baseline_row['median']

        max_accuracy_row = model_data.loc[model_data['median'].idxmax()]
        max_accuracy = max_accuracy_row['median']
        max_x = model_data['num_training_samples'].max()

        total_improvement = max_accuracy - baseline_y
        sample_efficiency = 0.0
        if max_x > 0 and baseline_x > 0 and max_x > baseline_x:
            log_diff = np.log10(max_x) - np.log10(baseline_x)
            if log_diff > 1e-9: sample_efficiency = total_improvement / log_diff
            else: sample_efficiency = np.inf
        elif max_x > 0 and baseline_x <= 0: sample_efficiency = total_improvement / np.log10(max_x + 1)

        plateau_samples, plateau_accuracy = None, None
        improvement_threshold = 0.01
        for i in range(1, len(model_data)):
            curr_row, prev_row = model_data.iloc[i], model_data.iloc[i-1]
            curr_x, prev_x = curr_row['num_training_samples'], prev_row['num_training_samples']
            curr_y, prev_y = curr_row['median'], prev_row['median']
            if prev_y is not None and prev_y != 0 and curr_x > prev_x:
                rel_improvement = (curr_y - prev_y) / abs(prev_y)
                if rel_improvement < improvement_threshold:
                    plateau_samples, plateau_accuracy = prev_x, prev_y
                    break

        metrics[group] = {
            'baseline_accuracy': baseline_y, 'max_accuracy': max_accuracy,
            'total_improvement': total_improvement,
            'relative_improvement': (total_improvement / abs(baseline_y) * 100) if baseline_y != 0 else np.inf,
            'sample_efficiency': sample_efficiency, 'plateau_samples': plateau_samples,
            'plateau_accuracy': plateau_accuracy
        }

        if total_improvement > 0:
            for _, row in model_data[model_data['num_training_samples'] >= baseline_x].iterrows():
                sample_size, current_gain = row['num_training_samples'], row['median'] - baseline_y
                pct_of_max_gain = (current_gain / total_improvement) * 100
                try: x_label = int(sample_size)
                except: x_label = str(sample_size)
                metrics[group][f'pct_gain_at_{x_label}'] = pct_of_max_gain
        else:
             for _, row in model_data[model_data['num_training_samples'] >= baseline_x].iterrows():
                 try: x_label = int(row['num_training_samples'])
                 except: x_label = str(row['num_training_samples'])
                 metrics[group][f'pct_gain_at_{x_label}'] = 0.0
    return metrics

def add_sample_counts_to_figure(fig, df, x_col, y_col, color_col=None):
    """Add sample count annotations."""
    if x_col not in df.columns or y_col not in df.columns: return fig
    df_copy = df.copy()
    df_copy[y_col] = pd.to_numeric(df_copy[y_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[x_col, y_col])
    if df_copy.empty: return fig

    try:
        group_cols = [x_col]
        if color_col and color_col != "None" and color_col in df_copy.columns:
            group_cols.append(color_col)

        counts = df_copy.groupby(group_cols).size().reset_index(name='count')
        max_y_values = df_copy.groupby(group_cols)[y_col].max().reset_index()
        counts = pd.merge(counts, max_y_values, on=group_cols, how='left')

        # Fallback for positioning if max_y fails for a group
        overall_max_y = df_copy.groupby(x_col)[y_col].max().reset_index().rename(columns={y_col: '_overall_max_y'})
        counts = pd.merge(counts, overall_max_y, on=x_col, how='left')
        counts[y_col] = counts[y_col].fillna(counts['_overall_max_y'])
        counts = counts.drop(columns=['_overall_max_y']).dropna(subset=[y_col])

        for _, row in counts.iterrows():
            fig.add_annotation(x=row[x_col], y=row[y_col], text=f'n={row["count"]}', showarrow=False, yshift=10, font=dict(size=9, color='grey'))
    except Exception as e:
        st.warning(f"Could not add sample count annotations: {e}")
    return fig

def create_and_update_plot(fig, df, x_label, y_label, color_label, facet_row, facet_col, plot_type, key_suffix, current_colors):
    """Update plot with formatting and display it (enhanced UI)."""
    clean_x_label = clean_label(x_label)
    clean_y_label = clean_label(y_label)
    clean_color_label = clean_label(color_label) if color_label and color_label != "None" else ""
    clean_facet_row = clean_label(facet_row) if facet_row and facet_row != "None" else ""
    clean_facet_col = clean_label(facet_col) if facet_col and facet_col != "None" else ""

    default_title = f"{clean_y_label} vs {clean_x_label}"
    if clean_color_label: default_title += f" by {clean_color_label}"
    if clean_facet_row or clean_facet_col:
        facet_dims = [f"Rows: {clean_facet_row}"] if clean_facet_row else []
        if clean_facet_col: facet_dims.append(f"Cols: {clean_facet_col}")
        default_title += f" ({', '.join(facet_dims)})"

    # Initialize customization variables with defaults
    custom_settings = {}

    with st.expander("🎨 Customize Plot Appearance", expanded=False):
        cust_tabs = st.tabs(["Layout & Axes", "Colors & Style", "Reference Lines", "Annotations"])

        with cust_tabs[0]:
            st.markdown("#### General Layout")
            col1, col2 = st.columns(2)
            with col1:
                custom_settings['plot_title'] = st.text_input("Plot Title", value=default_title, key=f"title_{key_suffix}")
                custom_settings['title_font_size'] = st.slider("Title Font Size", 10, 40, 18, key=f"title_font_{key_suffix}")
                custom_settings['plot_height'] = st.slider("Plot Height (px)", 300, 1500, 600, 50, key=f"height_{key_suffix}")
            with col2:
                custom_settings['show_legend'] = st.checkbox("Show Legend", True, key=f"legend_{key_suffix}")
                custom_settings['legend_font_size'] = st.slider("Legend Font Size", 6, 24, 10, key=f"legend_size_{key_suffix}", disabled=not custom_settings['show_legend'])
                custom_settings['legend_title_size'] = st.slider("Legend Title Size", 8, 26, 12, key=f"legend_title_size_{key_suffix}", disabled=not custom_settings['show_legend'])

            st.markdown("#### Axes Configuration")
            col1, col2 = st.columns(2)
            with col1:
                custom_settings['axis_title_size'] = st.slider("Axis Title Size", 8, 30, 14, key=f"axis_title_size_{key_suffix}")
                custom_settings['axis_tick_size'] = st.slider("Axis Tick Size", 6, 24, 12, key=f"axis_tick_size_{key_suffix}")
                custom_settings['y_tick_step'] = st.number_input("Y-axis Tick Step Size", min_value=0.0, max_value=10.0, value=0.1, step=0.01, help="Interval between y-axis tick marks. Set to 0 for auto.", key=f"y_tick_step_{key_suffix}")

            with col2:
                custom_settings['y_range_auto'] = st.checkbox("Auto Y-axis Range", True, key=f"y_auto_{key_suffix}")
                y_min_val, y_max_val = 0.0, 1.0
                # Ensure y_label exists and is numeric before calculating min/max
                if y_label in df.columns and pd.api.types.is_numeric_dtype(df[y_label]):
                    y_numeric = df[y_label].dropna()
                    if not y_numeric.empty: y_min_val, y_max_val = float(y_numeric.min()), float(y_numeric.max())
                custom_settings['y_min'] = st.number_input("Y-axis Min", value=y_min_val, key=f"ymin_{key_suffix}", disabled=custom_settings['y_range_auto'])
                custom_settings['y_max'] = st.number_input("Y-axis Max", value=y_max_val, key=f"ymax_{key_suffix}", disabled=custom_settings['y_range_auto'])
                custom_settings['y_range'] = None if custom_settings['y_range_auto'] else [custom_settings['y_min'], custom_settings['y_max']]

                custom_settings['x_range_auto'] = st.checkbox("Auto X-axis Range", True, key=f"x_auto_{key_suffix}")
                x_min_default, x_max_default = 0.0, 1.0
                # Ensure x_label exists and is numeric before calculating min/max
                if x_label in df.columns and pd.api.types.is_numeric_dtype(df[x_label]):
                    x_numeric = df[x_label].dropna()
                    if not x_numeric.empty: x_min_default, x_max_default = float(x_numeric.min()), float(x_numeric.max())
                custom_settings['x_min'] = st.number_input("X-axis Min", value=x_min_default, key=f"xmin_{key_suffix}", disabled=custom_settings['x_range_auto'])
                custom_settings['x_max'] = st.number_input("X-axis Max", value=x_max_default, key=f"xmax_{key_suffix}", disabled=custom_settings['x_range_auto'])
                custom_settings['x_range'] = None if custom_settings['x_range_auto'] else [custom_settings['x_min'], custom_settings['x_max']]

        with cust_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                custom_settings['theme'] = st.selectbox("Plot Theme", ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"], index=0, key=f"theme_{key_suffix}")
                custom_settings['show_grid'] = st.checkbox("Show Grid Lines", True, key=f"grid_{key_suffix}")
                if plot_type == 'box':
                    st.markdown("**Box Plot Options**")
                    custom_settings['box_points'] = st.selectbox("Show Points", ["outliers", "all", "suspectedoutliers", False], index=0, key=f"boxpoints_{key_suffix}")
                    custom_settings['box_notched'] = st.checkbox("Notched Boxes", False, key=f"boxnotch_{key_suffix}")
                    custom_settings['box_mode'] = st.selectbox("Box Mode", ["group", "overlay"], index=0, key=f"boxmode_{key_suffix}")

            with col2:
                color_options = {name: getattr(px.colors.qualitative, name) for name in px.colors.qualitative.__all__}
                color_options.update({name: getattr(px.colors.sequential, name) for name in px.colors.sequential.__all__ if not name.endswith("_r")})
                selected_palette_name = st.selectbox("Color Palette", list(color_options.keys()), index=list(color_options.keys()).index('Plotly') if 'Plotly' in color_options else 0, key=f"palette_{key_suffix}")
                selected_color_list = color_options.get(selected_palette_name, current_colors) # Fallback
                custom_settings['active_colors'] = selected_color_list

                st.write("Color Preview:")
                # FIX: Check if selected_color_list is iterable and not a function
                if isinstance(selected_color_list, Iterable) and not callable(selected_color_list):
                    color_preview_html = "".join([f'<span title="{color}" style="display:inline-block;width:20px;height:20px;background:{color};margin:1px;border:1px solid lightgrey;"></span>' for color in selected_color_list])
                    st.markdown(color_preview_html, unsafe_allow_html=True)
                else:
                     st.warning(f"Selected color palette '{selected_palette_name}' is not a valid list of colors.")
                     custom_settings['active_colors'] = current_colors # Revert

        with cust_tabs[2]: # Reference Lines
            st.markdown("Define reference lines (horizontal or vertical). Use 'mean' or 'median' as value to calculate from data.")
            ref_line_session_key = f"ref_lines_{key_suffix}"
            if ref_line_session_key not in st.session_state: st.session_state[ref_line_session_key] = []
            ref_lines_list = st.session_state[ref_line_session_key]
            indices_to_remove = []
            if not ref_lines_list: st.caption("No reference lines defined yet.")
            for i, line in enumerate(ref_lines_list):
                cols = st.columns([1, 2, 2, 1, 1, 1, 1])
                cols[0].text_input("Axis", value=line.get('axis','y').upper(), key=f"ref_axis_disp_{key_suffix}_{i}", disabled=True)
                cols[1].text_input("Value/Stat", value=str(line.get('value', '')), key=f"ref_val_disp_{key_suffix}_{i}", disabled=True)
                cols[2].text_input("Label", value=line.get('label', ''), key=f"ref_lab_disp_{key_suffix}_{i}", disabled=True)
                cols[3].markdown(f"<span style='background-color:{line.get('color', '#000000')}; padding: 2px 5px; border-radius: 3px; color: white; mix-blend-mode: difference;'>{line.get('color', '#000000')}</span>", unsafe_allow_html=True)
                cols[4].text_input("Style", value=line.get('style', 'dash'), key=f"ref_sty_disp_{key_suffix}_{i}", disabled=True)
                cols[5].text_input("Width", value=str(line.get('width', 1)), key=f"ref_wid_disp_{key_suffix}_{i}", disabled=True)
                if cols[6].button("❌", key=f"ref_rem_{key_suffix}_{i}", help="Remove this line"): indices_to_remove.append(i)
            if indices_to_remove:
                 for i in sorted(indices_to_remove, reverse=True): ref_lines_list.pop(i)
                 st.rerun()
            st.markdown("---"); st.markdown("##### Add New Reference Line")
            cols_new = st.columns([1, 2, 2, 1, 1, 1, 1])
            new_axis = cols_new[0].selectbox("Axis", ['y', 'x'], key=f"ref_axis_new_{key_suffix}")
            new_value = cols_new[1].text_input("Value or Stat ('mean', 'median')", key=f"ref_val_new_{key_suffix}")
            new_label = cols_new[2].text_input("Label (optional)", key=f"ref_lab_new_{key_suffix}")
            new_color = cols_new[3].color_picker("Color", "#888888", key=f"ref_col_new_{key_suffix}")
            new_style = cols_new[4].selectbox("Style", ['solid', 'dash', 'dot', 'dashdot'], index=1, key=f"ref_sty_new_{key_suffix}")
            new_width = cols_new[5].number_input("Width", min_value=1, max_value=10, value=1, step=1, key=f"ref_wid_new_{key_suffix}")
            if cols_new[6].button("➕ Add", key=f"ref_add_{key_suffix}"):
                if new_value:
                    is_stat = new_value.lower() in ['mean', 'median']
                    numeric_val = None
                    if not is_stat:
                        try: numeric_val = float(new_value)
                        except ValueError: st.error(f"Invalid value '{new_value}'. Must be numeric or 'mean'/'median'."); new_value = None
                    if new_value:
                         ref_lines_list.append({'axis': new_axis, 'value': numeric_val if not is_stat else new_value.lower(), 'label': new_label, 'color': new_color, 'style': new_style, 'width': new_width, 'position': 'top right'})
                         st.rerun()
                else: st.warning("Please provide a value or statistic.")

        with cust_tabs[3]:
            custom_settings['show_sample_counts'] = st.checkbox("Show Sample Counts (n=...)", value=False, key=f"show_counts_{key_suffix}")
            custom_settings['add_custom_annotation'] = st.checkbox("Add Custom Text Annotation", False, key=f"custom_annot_chk_{key_suffix}")
            if custom_settings['add_custom_annotation']:
                custom_settings['annot_text'] = st.text_input("Annotation Text", "My Note", key=f"annot_text_{key_suffix}")
                custom_settings['annot_x'] = st.number_input("X Position (numeric)", value=0.5, key=f"annot_x_{key_suffix}") # Simplified
                custom_settings['annot_y'] = st.number_input("Y Position", value=0.5, key=f"annot_y_{key_suffix}")
                custom_settings['annot_arrow'] = st.checkbox("Show Arrow", True, key=f"annot_arrow_{key_suffix}")
                custom_settings['annot_color'] = st.color_picker("Text Color", "#000000", key=f"annot_color_{key_suffix}")
                custom_settings['annot_size'] = st.slider("Text Size", 8, 24, 12, key=f"annot_size_{key_suffix}")

    # --- Apply Customizations & Display ---
    try:
        # Use .get with defaults for all custom_settings lookups
        fig.update_layout(template=custom_settings.get('theme', 'plotly_white'))
        fig.update_layout(
            title={'text': custom_settings.get('plot_title', default_title), 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=custom_settings.get('title_font_size', 18))},
            height=custom_settings.get('plot_height', 600),
            showlegend=custom_settings.get('show_legend', True),
            xaxis_title=clean_x_label, yaxis_title=clean_y_label,
            xaxis_range=custom_settings.get('x_range'), yaxis_range=custom_settings.get('y_range'),
            legend=dict(title=dict(text=clean_color_label, font=dict(size=custom_settings.get('legend_title_size', 12))), font=dict(size=custom_settings.get('legend_font_size', 10)), bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
            colorway=custom_settings.get('active_colors', current_colors), # Use fetched colors
            boxmode=custom_settings.get('box_mode', 'group') if plot_type == 'box' else None
        )
        fig.update_xaxes(showgrid=custom_settings.get('show_grid', True), title_font=dict(size=custom_settings.get('axis_title_size', 14)), tickfont=dict(size=custom_settings.get('axis_tick_size', 12)))
        # Handle y_tick_step=0 case for auto ticks
        y_dtick = custom_settings.get('y_tick_step', 0.1)
        fig.update_yaxes(showgrid=custom_settings.get('show_grid', True), title_font=dict(size=custom_settings.get('axis_title_size', 14)), tickfont=dict(size=custom_settings.get('axis_tick_size', 12)), dtick=y_dtick if y_dtick > 0 else None)

        if plot_type == 'box': fig.update_traces(boxpoints=custom_settings.get('box_points', 'outliers'), notched=custom_settings.get('box_notched', False), selector=dict(type='box'))

        active_ref_lines = st.session_state.get(f"ref_lines_{key_suffix}", [])
        if active_ref_lines:
            target_df_ref = df.copy()
            y_col_ref, x_col_ref = y_label, x_label
            if y_label == '_metric_value_' and '_metric_value_' in target_df_ref.columns: y_col_ref = '_metric_value_'
            if x_label == '_metric_variable_' and '_metric_variable_' in target_df_ref.columns: x_col_ref = '_metric_variable_'
            fig = add_reference_lines_generic(fig, target_df_ref, x_col_ref, y_col_ref, active_ref_lines)

        if custom_settings.get('show_sample_counts', False):
             y_col_counts = y_label if y_label in df.columns and pd.api.types.is_numeric_dtype(df[y_label]) else '_metric_value_' if '_metric_value_' in df.columns else None
             if y_col_counts: fig = add_sample_counts_to_figure(fig, df, x_label, y_col_counts, color_label)
             else: st.warning(f"Cannot add sample counts: Y-axis column '{y_label}' not valid.")

        if custom_settings.get('add_custom_annotation', False) and custom_settings.get('annot_text'):
            fig.add_annotation(
                x=custom_settings.get('annot_x', 0.5), y=custom_settings.get('annot_y', 0.5),
                text=custom_settings.get('annot_text', ''), showarrow=custom_settings.get('annot_arrow', True),
                arrowhead=1 if custom_settings.get('annot_arrow', True) else 0,
                font=dict(size=custom_settings.get('annot_size', 12), color=custom_settings.get('annot_color', "#000000")),
                align="left", bgcolor="rgba(255, 255, 255, 0.7)"
             )

    except Exception as e:
        st.error(f"Error applying plot customizations: {e}")
        st.code(traceback.format_exc())

    # Display Plot and Export Options
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_{key_suffix}")
    st.markdown("---")
    st.markdown("##### Export Plot")
    export_cols = st.columns(3)
    plot_filename_base = f"{clean_y_label}_vs_{clean_x_label}"
    if clean_color_label: plot_filename_base += f"_by_{clean_color_label}"
    plot_filename_base = "".join(c if c.isalnum() else "_" for c in plot_filename_base).lower()

    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)
        export_cols[0].download_button("Download PNG", png_bytes, f"{plot_filename_base}.png", "image/png", key=f"dl_png_{key_suffix}")
    except Exception as e: export_cols[0].warning(f"PNG export failed: {e}")
    try:
        html_buffer = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        export_cols[1].download_button("Download HTML", html_buffer, f"{plot_filename_base}.html", "text/html", key=f"dl_html_{key_suffix}")
    except Exception as e: export_cols[1].warning(f"HTML export failed: {e}")
    try:
        json_str = pio.to_json(fig)
        export_cols[2].download_button("Download JSON", json_str, f"{plot_filename_base}.json", "application/json", key=f"dl_json_{key_suffix}")
    except Exception as e: export_cols[2].warning(f"JSON export failed: {e}")


def add_reference_lines_generic(fig, df, x_col, y_col, ref_lines_config):
    """Adds reference lines (h or v) based on config (generic version)."""
    if not ref_lines_config or df is None or df.empty: return fig
    for line_config in ref_lines_config:
        axis = line_config.get('axis', 'y')
        value = line_config.get('value')
        label = line_config.get('label', '')
        color = line_config.get('color', 'grey')
        style = line_config.get('style', 'dash')
        width = line_config.get('width', 1)
        position = line_config.get('position', 'top right')
        target_val, target_col = None, y_col if axis == 'y' else x_col

        if target_col not in df.columns:
             st.warning(f"Reference line calculation skipped: Column '{target_col}' not found.")
             continue

        if isinstance(value, str) and value.lower() in ['mean', 'median']:
            # Ensure target column is numeric before calculating stats
            numeric_data = pd.to_numeric(df[target_col], errors='coerce').dropna()
            if not numeric_data.empty:
                target_val = numeric_data.mean() if value.lower() == 'mean' else numeric_data.median()
                label = f"{value.capitalize()} ({target_val:.3f})" if not label else f"{label} ({target_val:.3f})"
            else:
                 st.warning(f"Could not calculate {value} for reference line: Column '{target_col}' has no numeric data.")
                 continue
        else:
            try: target_val = float(value)
            except (ValueError, TypeError):
                st.warning(f"Invalid value '{value}' for reference line '{label}'. Must be numeric or 'mean'/'median'.")
                continue
        if target_val is not None:
            line_params = dict(line=dict(color=color, width=width, dash=style))
            annot_params = dict(annotation_text=label, annotation_position=position)
            try:
                if axis == 'y': fig.add_hline(y=target_val, **line_params, **annot_params)
                elif axis == 'x': fig.add_vline(x=target_val, **line_params, **annot_params)
            except Exception as e:
                 st.warning(f"Could not add reference line '{label}' at {target_val}: {e}")
    return fig

# MODIFICATION: Function now takes the uploaded file object
@st.cache_data # Cache the processing based on the uploaded file
def process_uploaded_data(uploaded_file_obj):
    """Loads and processes data from an uploaded file object."""
    if uploaded_file_obj is None:
        return None
    try:
        df = pd.read_csv(uploaded_file_obj, low_memory=False)
        st.sidebar.info(f"Processing {len(df)} raw records from uploaded file.")

        # --- Start of processing logic (similar to process_dataset) ---
        if 'file_exists' in df.columns:
            original_count = len(df)
            df = df[df['file_exists'] == True].copy()
            st.sidebar.info(f"Removed {original_count - len(df)} records where 'file_exists' was False.")

        if 'training_status' in df.columns: df['training_status'] = df['training_status'].astype(str)
        if 'model_name' in df.columns: df['model_name'] = df['model_name'].astype(str)
        for col in ['predicted_label', 'ground_truth_label']:
             if col in df.columns: df[col] = df[col].astype(str)

        original_count_dedup = len(df)
        df.drop_duplicates(inplace=True, keep='first')
        num_duplicates = original_count_dedup - len(df)
        if num_duplicates > 0: st.sidebar.info(f"Removed {num_duplicates} duplicate rows.")
        st.sidebar.info(f"Using {len(df)} unique records.")
        # --- End of processing logic ---

        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.code(traceback.format_exc())
        return None


# --- Sidebar Setup ---
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Configure data loading, filtering, and global plot settings.")
st.sidebar.divider()

# --- MODIFICATION: Data Loading via Uploader ---
st.sidebar.subheader("1. Load Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Question-Level Results CSV",
    type=["csv"],
    help="Upload the large CSV file containing the analysis results (e.g., downloaded from [Your Data Source Link])."
)

# Process the uploaded file (cached)
df_processed = process_uploaded_data(uploaded_file)

# --- Main App Logic ---
if df_processed is not None and not df_processed.empty:
    # Use session state to store the processed dataframe reference
    st.session_state.df_processed = df_processed

    # --- Sidebar Filters (Applied to df_processed) ---
    st.sidebar.divider()
    st.sidebar.subheader("2. Filter Data")
    st.sidebar.markdown("Select the specific configuration to analyze.")

    # REVISION: Single selection for Model Name
    model_options = sorted(df_processed['model_name'].unique())
    # Reset selected model if options change (e.g., new file uploaded)
    if 'select_model' not in st.session_state or st.session_state.select_model not in model_options:
        st.session_state.select_model = model_options[0] if model_options else None
    selected_model = st.sidebar.selectbox(
        "Select Model Name", options=model_options, key="select_model",
        help="Choose the model you want to analyze."
    )

    # REVISION: Single selection for Training Status
    training_options = sorted(df_processed['training_status'].unique())
    # Reset selected status if options change
    if 'select_training' not in st.session_state or st.session_state.select_training not in training_options:
         st.session_state.select_training = training_options[0] if training_options else None
    selected_training_status = st.sidebar.radio(
        "Select Training Status", options=training_options, key="select_training",
        horizontal=True, help="Analyze either the 'untrained' or 'trained' version."
    )

    # --- Apply Core Filters ---
    df_core_filtered = df_processed[
        (df_processed['model_name'] == selected_model) &
        (df_processed['training_status'] == selected_training_status)
    ].copy()

    st.sidebar.divider()
    st.sidebar.markdown("**Additional Filters**")

    # --- Additional Parameter Filters ---
    active_filters = {}
    gen_params_to_filter = ['generation', 'response_format', 'response_type', 'prompt_type', 'explanation_type']
    dataset_params_to_filter = ['dataset', 'eval_dataset']

    for param in gen_params_to_filter + dataset_params_to_filter:
        if param in df_core_filtered.columns:
            options = sorted(df_core_filtered[param].dropna().unique())
            if len(options) > 1:
                filter_key = f"filter_{param}"
                # Initialize session state for filter if it doesn't exist
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = options # Default to all selected
                # Ensure default is valid within current options
                current_default = [opt for opt in st.session_state[filter_key] if opt in options]
                if not current_default: # If previous default is no longer valid, reset
                    current_default = options
                selected_vals = st.sidebar.multiselect(f"{clean_label(param)}", options=options, key=filter_key, default=current_default)
                active_filters[param] = selected_vals

    # Apply additional filters
    df_final_filtered = df_core_filtered.copy()
    filter_summary = [f"Model: **{selected_model}**", f"Training: **{selected_training_status}**"]
    for col, selected_values in active_filters.items():
        # Only filter if not all options are selected
        if selected_values and len(selected_values) < len(df_core_filtered[col].dropna().unique()):
            df_final_filtered = df_final_filtered[df_final_filtered[col].isin(selected_values)]
            filter_summary.append(f"{clean_label(col)}: {', '.join(map(str, selected_values))}")

    st.sidebar.divider()
    st.sidebar.markdown(f"**Analysis Scope:** Analyzing **{len(df_final_filtered)}** records for:")
    for item in filter_summary: st.sidebar.markdown(f"- {item}")

    # --- Grouping Parameter Selection ---
    st.sidebar.subheader("Group By Parameter")
    st.sidebar.markdown("Select ONE parameter to group the bias analysis by.")
    potential_group_params = [
        col for col in ['generation', 'response_format', 'response_type', 'prompt_type', 'explanation_type', 'dataset', 'eval_dataset']
        if col in df_final_filtered.columns and df_final_filtered[col].nunique() > 1
    ]
    group_by_param = st.sidebar.selectbox(
        "Group Analysis By", options=["None"] + potential_group_params, index=0,
        format_func=lambda x: "Overall (No Grouping)" if x == "None" else clean_label(x),
        help="Optional: Group the bias results by the selected parameter."
    )

    # --- Plot Color Palette ---
    st.sidebar.subheader("Plot Settings")
    color_palette = st.sidebar.selectbox(
        "Color Palette", options=['Plotly', 'Set1', 'Set2', 'Set3','Alphabet','D3', 'Bold', 'Prism', 'Dark24', 'Light24'], index=0,
        key="global_color_palette", help="Select color scheme for plots."
    )
    try: colors = getattr(px.colors.qualitative, color_palette)
    except AttributeError: st.sidebar.warning(f"Palette {color_palette} not found, using Plotly."); colors = px.colors.qualitative.Plotly
    pio.templates.default = "plotly_white"

    # --- Main Analysis Area ---
    if not df_final_filtered.empty:
        st.markdown("### Current Analysis Configuration")
        st.markdown(f"**Model:** `{selected_model}` | **Training Status:** `{selected_training_status}`")
        if group_by_param != "None": st.markdown(f"**Grouped By:** `{clean_label(group_by_param)}`")
        else: st.markdown("**Grouped By:** `Overall (No Grouping)`")
        other_filters_applied = [f"{clean_label(k)}: {', '.join(map(str, v))}" for k, v in active_filters.items() if v and len(v) < len(df_core_filtered[k].dropna().unique())]
        if other_filters_applied: st.markdown(f"**Additional Filters:** `{'; '.join(other_filters_applied)}`")
        st.divider()

        # --- Tabbed Analysis ---
        tabs = st.tabs(["📊 Option Selection Bias", "❓ Label Confusion Analysis"]) # REVISION: Removed tabs

        # ================= Tab 1: Option Selection Bias =================
        with tabs[0]:
            st.header("Option Selection Bias")
            st.markdown("Compares the distribution of the model's predicted answers (A, B, C, D) against the distribution of the correct answers.")

            # --- Overall Bias Calculation ---
            pred_option_counts_overall = df_final_filtered['predicted_label'].value_counts(normalize=True) * 100
            gt_option_counts_overall = df_final_filtered['ground_truth_label'].value_counts(normalize=True) * 100
            all_options = sorted(list(set(pred_option_counts_overall.index) | set(gt_option_counts_overall.index)))
            option_bias_data_overall = []
            for option in all_options:
                pred_pct = pred_option_counts_overall.get(option, 0)
                gt_pct = gt_option_counts_overall.get(option, 0)
                option_bias_data_overall.append({'Option': option, 'Predicted (%)': pred_pct, 'Ground Truth (%)': gt_pct, 'Difference (%)': pred_pct - gt_pct})
            option_bias_df_overall = pd.DataFrame(option_bias_data_overall).sort_values('Option')
            rmse_overall = np.sqrt(np.mean(option_bias_df_overall['Difference (%)']**2))

            # --- Grouped Analysis ---
            if group_by_param != "None":
                st.subheader(f"Bias Analysis Grouped by {clean_label(group_by_param)}")
                try:
                    grouped_data = df_final_filtered.groupby(group_by_param)
                    bias_metrics_grouped = []
                    all_group_option_data = []
                    for name, group in grouped_data:
                        pred_counts_group = group['predicted_label'].value_counts(normalize=True) * 100
                        gt_counts_group = group['ground_truth_label'].value_counts(normalize=True) * 100
                        group_bias = {}; group_option_data = []; sum_sq_diff = 0
                        for option in all_options:
                            pred_pct = pred_counts_group.get(option, 0); gt_pct = gt_counts_group.get(option, 0); diff = pred_pct - gt_pct
                            sum_sq_diff += diff**2; group_bias[f'Bias_{option}'] = diff
                            group_option_data.append({'Group': name, 'Option': option, 'Percentage': pred_pct, 'Type': 'Predicted'})
                            group_option_data.append({'Group': name, 'Option': option, 'Percentage': gt_pct, 'Type': 'Ground Truth'})
                        rmse_group = np.sqrt(sum_sq_diff / len(all_options))
                        bias_metrics_grouped.append({ 'Group': name, 'RMSE_Bias': rmse_group, **group_bias })
                        all_group_option_data.extend(group_option_data)
                    bias_df_grouped = pd.DataFrame(bias_metrics_grouped)
                    all_group_option_df = pd.DataFrame(all_group_option_data)

                    fig_options_grouped = px.bar(all_group_option_df, x='Group', y='Percentage', color='Type', barmode='group', facet_col='Option', category_orders={"Option": all_options}, title=f"Predicted vs Ground Truth Option Distribution by {clean_label(group_by_param)}", labels={'Group': clean_label(group_by_param), 'Percentage': 'Percentage (%)'}, color_discrete_map={'Predicted': colors[0], 'Ground Truth': colors[1]})
                    fig_options_grouped.update_xaxes(tickangle=45); st.plotly_chart(fig_options_grouped, use_container_width=True)
                    fig_bias_grouped = px.bar(bias_df_grouped.sort_values('RMSE_Bias', ascending=False), x='Group', y='RMSE_Bias', title=f"Overall Option Selection Bias (RMSE) by {clean_label(group_by_param)}", labels={'Group': clean_label(group_by_param), 'RMSE_Bias': 'RMSE Bias (%)'}, color='RMSE_Bias', color_continuous_scale='Reds')
                    fig_bias_grouped.update_xaxes(tickangle=45); st.plotly_chart(fig_bias_grouped, use_container_width=True)
                    st.dataframe(bias_df_grouped.style.format({'RMSE_Bias': '{:.2f}', **{f'Bias_{opt}': '{:.2f}' for opt in all_options}}), use_container_width=True)
                except Exception as e: st.warning(f"Could not perform grouped bias analysis: {e}")

            # --- Overall Bias Display ---
            st.subheader("Overall Option Selection Bias (Across Filtered Data)")
            col1, col2 = st.columns([2,1])
            with col1:
                combined_overall_counts = pd.concat([option_bias_df_overall[['Option', 'Predicted (%)']].rename(columns={'Predicted (%)':'Percentage'}).assign(Type='Predicted'), option_bias_df_overall[['Option', 'Ground Truth (%)']].rename(columns={'Ground Truth (%)':'Percentage'}).assign(Type='Ground Truth')])
                fig_options_overall = px.bar(combined_overall_counts, x='Option', y='Percentage', color='Type', barmode='group', title="Overall Distribution of Options (A, B, C, D)", labels={'Percentage': 'Percentage (%)'}, color_discrete_map={'Predicted': colors[0], 'Ground Truth': colors[1]})
                st.plotly_chart(fig_options_overall, use_container_width=True)
            with col2:
                 st.metric("Overall Option Selection Bias (RMSE)", f"{rmse_overall:.2f}%", help="Lower values indicate less overall bias.")
                 st.dataframe(option_bias_df_overall.style.format({'Predicted (%)': '{:.2f}', 'Ground Truth (%)': '{:.2f}', 'Difference (%)': '{:.2f}'}), use_container_width=True)

            st.divider()
            # --- Option Selection Bias by True Label ---
            st.subheader("Option Selection Bias by Ground Truth Label")
            st.markdown("Analyzes if the model prefers predicting certain options when the *actual* correct answer is A, B, C, or D.")
            try:
                pred_label_option_pivot = pd.crosstab(df_final_filtered['ground_truth_label'], df_final_filtered['predicted_label'], normalize='index') * 100
                all_options_labels = ['A', 'B', 'C', 'D']
                pred_label_option_pivot = pred_label_option_pivot.reindex(index=all_options_labels, columns=all_options_labels, fill_value=0)
                bias_label_option_pivot = pred_label_option_pivot.copy()
                for idx in bias_label_option_pivot.index: bias_label_option_pivot.loc[idx, idx] -= 100

                bias_tabs = st.tabs(["Predicted Distribution Heatmap", "Bias Heatmap", "Bias by Label (RMSE)"])
                with bias_tabs[0]:
                    fig_pred_heatmap = px.imshow(pred_label_option_pivot, labels=dict(x="Predicted Option", y="Ground Truth Label", color="Prediction %"), x=all_options_labels, y=all_options_labels, color_continuous_scale="Blues", title="Heatmap: What the Model Predicts Given the True Answer")
                    fig_pred_heatmap.update_layout(coloraxis_colorbar=dict(title="%")); st.plotly_chart(fig_pred_heatmap, use_container_width=True)
                with bias_tabs[1]:
                    fig_bias_heatmap = px.imshow(bias_label_option_pivot, labels=dict(x="Predicted Option", y="Ground Truth Label", color="Bias %"), x=all_options_labels, y=all_options_labels, color_continuous_scale="RdBu", color_continuous_midpoint=0, title="Heatmap: Option Prediction Bias Given the True Answer")
                    fig_bias_heatmap.update_layout(coloraxis_colorbar=dict(title="Bias %")); st.plotly_chart(fig_bias_heatmap, use_container_width=True)
                with bias_tabs[2]:
                    rmse_by_label = bias_label_option_pivot.apply(lambda row: np.sqrt(np.mean(row**2)), axis=1).reset_index()
                    rmse_by_label.columns = ['Ground Truth Label', 'RMSE_Bias']
                    fig_rmse_bar = px.bar(rmse_by_label.sort_values('RMSE_Bias', ascending=False), x='Ground Truth Label', y='RMSE_Bias', title="Overall Bias (RMSE) for Each Ground Truth Label", labels={'RMSE_Bias': 'RMSE Bias (%)'}, color='RMSE_Bias', color_continuous_scale='Reds')
                    st.plotly_chart(fig_rmse_bar, use_container_width=True)
            except Exception as e: st.warning(f"Could not generate bias by label analysis: {e}")


        # ================= Tab 2: Label Confusion Analysis =================
        with tabs[1]:
            st.header("Label Confusion Analysis")
            st.markdown("Shows how often the model predicts one label when the ground truth is another.")

            # --- Overall Confusion Matrix ---
            st.subheader("Overall Label Confusion Matrix (Across Filtered Data)")
            try:
                label_confusion = pd.crosstab(df_final_filtered['ground_truth_label'], df_final_filtered['predicted_label'], normalize='index') * 100
                all_options_labels = ['A', 'B', 'C', 'D'] # Define expected labels
                label_confusion = label_confusion.reindex(index=all_options_labels, columns=all_options_labels, fill_value=0)
                fig_label_confusion = px.imshow(label_confusion, labels=dict(x="Predicted Label", y="Ground Truth Label", color="Prediction %"), x=all_options_labels, y=all_options_labels, color_continuous_scale="Blues", title="Confusion Matrix: % of Times Each Label Was Predicted Given the Ground Truth")
                fig_label_confusion.update_layout(coloraxis_colorbar=dict(title="%")); st.plotly_chart(fig_label_confusion, use_container_width=True)
            except Exception as e: st.warning(f"Could not generate overall confusion matrix: {e}")

            # --- Grouped Confusion Analysis ---
            if group_by_param != "None":
                st.subheader(f"Label Accuracy by {clean_label(group_by_param)}")
                st.markdown(f"Shows the percentage of correct predictions for each ground truth label, broken down by **{clean_label(group_by_param)}**.")
                try:
                    df_final_filtered['label_match'] = (df_final_filtered['ground_truth_label'] == df_final_filtered['predicted_label']).astype(int)
                    label_acc_by_group = df_final_filtered.groupby([group_by_param, 'ground_truth_label'])['label_match'].mean().reset_index()
                    label_acc_by_group['Accuracy (%)'] = label_acc_by_group['label_match'] * 100
                    fig_label_acc_grouped = px.bar(label_acc_by_group, x='ground_truth_label', y='Accuracy (%)', color=group_by_param, title=f"Label Prediction Accuracy by {clean_label(group_by_param)}", labels={'ground_truth_label': 'Ground Truth Label'}, barmode='group', category_orders={"ground_truth_label": all_options_labels}, color_discrete_sequence=colors)
                    st.plotly_chart(fig_label_acc_grouped, use_container_width=True)
                except Exception as e: st.warning(f"Could not generate grouped label accuracy plot: {e}")

    # --- Display Raw Data ---
    st.divider()
    with st.expander("View Filtered Data Sample"):
        st.markdown(f"Showing the first 100 rows of the **{len(df_final_filtered)}** currently filtered data points.")
        st.dataframe(df_final_filtered.head(100))

else:
    st.warning("Please upload a CSV data file using the sidebar to begin analysis.")
