# plotting.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from constants import (
    SUN_MARKER_SIZE,
    LOCAL_NEIGHBORHOOD_OPACITY,
    LEGEND_STYLE,
    CAMERA_DEFAULT_EYE_X,
    CAMERA_DEFAULT_EYE_Y,
    CAMERA_DEFAULT_EYE_Z,
    CAMERA_EYE_FACTOR,
    CAMERA_Z_FACTOR,
    CAMERA_EFFECTIVE_DISTANCE_MIN,
    CAMERA_EFFECTIVE_DISTANCE_MAX,
    CLASSIFICATION_COLORS,  # Make sure this is imported if not already
    HOVER_FONT_COLOR,
    HOVER_BG_COLORS,
)

from data_processing import format_value


def create_sphere_surface(
    radius,
    color,
    opacity=0.1,
    name="Reference Sphere",
    resolution=20,
    showlegend=True,
    visible=True,
):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale=[[0, color], [1, color]],
        surfacecolor=np.full_like(x_sphere, 0.5),
        cmin=0,
        cmax=1,
        showscale=False,
        opacity=opacity,
        name=name,
        hoverinfo="name",
        showlegend=showlegend,
        visible=visible,
    )


def create_base_figure(df, neighborhood_sphere_initial_visibility=True):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=SUN_MARKER_SIZE, color="yellow", opacity=1.0),
            name="Sol (Sun)",
            hoverinfo="name",
            showlegend=True,
            visible=True,
        )
    )
    neighborhood_radius_ly = 1000
    fig.add_trace(
        create_sphere_surface(
            radius=neighborhood_radius_ly,
            color="deepskyblue",
            opacity=LOCAL_NEIGHBORHOOD_OPACITY,
            name=f"Local Neighborhood (~{neighborhood_radius_ly} ly)",
            showlegend=True,
            visible=neighborhood_sphere_initial_visibility,
        )
    )

    camera_eye_x, camera_eye_y, camera_eye_z = (
        CAMERA_DEFAULT_EYE_X,
        CAMERA_DEFAULT_EYE_Y,
        CAMERA_DEFAULT_EYE_Z,
    )
    if df is not None and not df.empty and "sy_dist_ly" in df.columns:
        max_dist = df["sy_dist_ly"].max()
        if pd.notna(max_dist) and max_dist > 0:
            effective_distance = np.clip(
                max_dist, CAMERA_EFFECTIVE_DISTANCE_MIN, CAMERA_EFFECTIVE_DISTANCE_MAX
            )
            camera_eye_x = effective_distance * CAMERA_EYE_FACTOR
            camera_eye_y = effective_distance * CAMERA_EYE_FACTOR
            camera_eye_z = effective_distance * CAMERA_Z_FACTOR

    fig.update_layout(
        scene=dict(
            xaxis_title="X (ly)",
            yaxis_title="Y (ly)",
            zaxis_title="Z (ly)",
            xaxis=dict(
                gridcolor="dimgray",
                zerolinecolor="dimgray",
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)",
            ),
            yaxis=dict(
                gridcolor="dimgray",
                zerolinecolor="dimgray",
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)",
            ),
            zaxis=dict(
                gridcolor="dimgray",
                zerolinecolor="dimgray",
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)",
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=camera_eye_x, y=camera_eye_y, z=camera_eye_z),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=LEGEND_STYLE,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def display_exoplanet_classification_map(df_input):
    st.header("Exoplanet Classification Map")
    st.markdown(
        "3D visualization of exoplanetary systems, colored by their habitability classification. Hover over planets for detailed information."
    )

    if df_input is None or df_input.empty:
        st.warning("No exoplanet data available to display.")
        return

    df = df_input.copy()
    fig = create_base_figure(df, neighborhood_sphere_initial_visibility="legendonly")

    for category, color in CLASSIFICATION_COLORS.items():
        category_df = df[df["classification_category"] == category]
        if not category_df.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=category_df["x"],
                    y=category_df["y"],
                    z=category_df["z"],
                    mode="markers",
                    marker=dict(
                        size=category_df["marker_size"],
                        color=color,
                        opacity=0.9,
                    ),
                    text=category_df["hover_text_main"],
                    hoverinfo="text",
                    name=category,
                    hoverlabel=dict(
                        bgcolor=HOVER_BG_COLORS.get(category, "grey"),
                        font=dict(color=HOVER_FONT_COLOR),
                        bordercolor="rgba(0,0,0,0.6)",
                        namelength=-1,
                    ),
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=category,
                )
            )

    fig.update_layout(title_text="Exoplanetary Systems: Habitability Classification")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def display_habitability_analysis_dashboard(
    df, param_map, thresholds, classification_colors
):
    st.header("Habitability Analysis & Insights")
    if df is None or df.empty:
        st.warning("No data available for analysis dashboard.")
        return

    st.subheader("Classification Summary")
    classification_counts = df["classification_category"].value_counts().sort_index()
    classification_percentages = (
        df["classification_category"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .sort_index()
    )
    summary_df = pd.DataFrame(
        {
            "Category": classification_counts.index,
            "Count": classification_counts.values,
            "Percentage (%)": classification_percentages.values,
        }
    )
    st.table(summary_df)
    st.write("*Earth is included in the Excellent Candidate Category")

    st.subheader("Key Planets & Findings Overview")
    earth_row = df[(df["pl_name"] == "Earth") & (df["hostname"] == "Sol")]
    kepler22b_row = df[df["pl_name"] == "Kepler-22 b"]

    cols_key = st.columns(2)
    with cols_key[0]:
        st.markdown("**Earth (Sol):**")
        if not earth_row.empty:
            st.markdown(
                f"- Classification: **{earth_row['classification_category'].iloc[0]}**"
            )
            st.markdown(
                f"- Serves as the primary validation standard, classified as 'Excellent Candidate'."
            )
        else:
            st.markdown("- Earth not found or not standardized in dataset.")

    with cols_key[1]:
        st.markdown("**Kepler-22 b:**")
        if not kepler22b_row.empty:
            st.markdown(
                f"- Classification: **{kepler22b_row['classification_category'].iloc[0]}**"
            )
            st.markdown(
                f"- A notable Earth analog. Its classification depends on meeting all 8 parameter thresholds."
            )
            if (
                kepler22b_row["classification_category"].iloc[0]
                != "Excellent Candidate"
            ):
                st.markdown(
                    f"  _Note: Current classification is '{kepler22b_row['classification_category'].iloc[0]}'. This might be due to data values in this CSV or missing parameters for full assessment._"
                )
        else:
            st.markdown("- Kepler-22 b not found by that exact name.")

    st.markdown("**M-dwarf Systems ('Good Planet, Poor Star'):**")
    st.markdown(
        "Planets in this category, often orbiting M-dwarfs, show suitable planetary conditions but marginal stellar conditions according to the defined thresholds, representing important alternative habitability scenarios."
    )

    st.subheader("Parameter Distributions by Classification")
    st.markdown(
        "Histograms for the 8 key parameters. Earth's value is marked (red dashed line), and habitability thresholds are shaded (green)."
    )

    num_cols_hist = 2
    param_keys = list(param_map.keys())
    for i in range(0, len(param_keys), num_cols_hist):
        cols = st.columns(num_cols_hist)
        for j in range(num_cols_hist):
            if i + j < len(param_keys):
                param_col_name = param_keys[i + j]
                with cols[j]:
                    param_details = param_map[param_col_name]
                    param_threshold_values = thresholds[param_col_name]

                    fig_hist = px.histogram(
                        df,
                        x=param_col_name,
                        color="classification_category",
                        marginal="rug",
                        color_discrete_map=classification_colors,
                        labels={
                            param_col_name: f"{param_details['name']} ({param_details['unit']})"
                        },
                        opacity=0.7,
                        barmode="overlay",
                    )
                    fig_hist.update_layout(
                        title_text=f"{param_details['name']}",
                        xaxis_title_text=f"{param_details['name']} ({param_details['unit']})",
                        yaxis_title_text="Count",
                        legend_title_text="Classification",
                    )
                    if not earth_row.empty and pd.notna(
                        earth_row[param_col_name].iloc[0]
                    ):
                        earth_val = earth_row[param_col_name].iloc[0]
                        fig_hist.add_vline(
                            x=earth_val,
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Earth",
                            annotation_position="top right",
                        )
                    fig_hist.add_vrect(
                        x0=param_threshold_values["min"],
                        x1=param_threshold_values["max"],
                        fillcolor="green",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                        annotation_text="Habitable Range",
                        annotation_position="top left",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Summary of Key Insights")
    good_star_poor_planet_percentage = classification_percentages.get(
        "Good Star, Poor Planet", 0
    )
    st.markdown(
        f"**Detection Bias:** Approximately **{good_star_poor_planet_percentage:.1f}%** of classified exoplanets fall into the 'Good Star, Poor Planet' category, indicating their host stars are suitable but the planets are not. This highlights observational biases towards easier-to-detect, often less habitable, planets."
    )

    st.markdown(
        "**Rarity of Earth-like Conditions:** Earth's statistical position (e.g., ~70th percentile for unusualness) suggests Earth-like conditions are uncommon but achievable. Only a small fraction (0.6%) are 'Excellent Candidates', emphasizing their rarity while acknowledging that detection bias inflates this perceived scarcity."
    )


# MODIFIED FUNCTION: display_cluster_visualization
def display_cluster_visualization(
    base_df, run_clustering_func, feature_options_map
):  # Added feature_options_map
    st.header("K-Means Cluster Analysis")
    st.markdown(
        "Explore exoplanet clusters based on selected physical parameters. This is an unsupervised learning approach to find natural groupings in the data."
    )

    if base_df is None or base_df.empty:  # Added check for base_df empty as well
        st.warning("No exoplanet data available for K-Means Clustering.")
        return

    if not feature_options_map:  # Check if the map itself is empty
        st.sidebar.error(
            "Feature options for clustering are not available. Cannot proceed with K-Means."
        )
        st.error(
            "Configuration error: Feature options map is missing for K-Means clustering."
        )
        return

    # Default selection uses the friendly names (keys of the map)
    default_selection_friendly = ["Planet Radius (ER)", "Equilibrium Temperature (K)"]

    # Ensure default selections are valid keys in the provided feature_options_map
    valid_default_selection = [
        f for f in default_selection_friendly if f in feature_options_map
    ]
    if (
        not valid_default_selection and feature_options_map
    ):  # If defaults are bad, pick first few available
        valid_default_selection = list(feature_options_map.keys())[:2]

    selected_features_friendly = st.sidebar.multiselect(
        "Select features for K-Means clustering:",
        options=list(feature_options_map.keys()),  # Use keys from the passed map
        default=valid_default_selection,
        key="cluster_features_multiselect",
    )

    # Map friendly names back to actual DataFrame column names
    selected_features_actual = [
        feature_options_map[f]
        for f in selected_features_friendly
        if f in feature_options_map
    ]

    if not selected_features_actual:
        st.sidebar.warning("Please select at least one feature for clustering.")
        # Optionally, you could prevent further execution or display a message in the main area
        # st.info("Select features in the sidebar to perform K-Means clustering.")
        # return # Or let run_clustering handle it if it's robust to empty feature list

    min_clusters = 1
    # Calculate max_clusters_possible based on non-NaN rows for *at least one* selected feature if features are selected
    # This is a bit complex to do perfectly here, so we'll stick to len(base_df) as a simpler upper bound for the slider.
    # A more precise count would involve dropping NaNs for selected_features_actual from base_df.
    max_clusters_possible = (
        len(base_df) if base_df is not None and not base_df.empty else 1
    )

    slider_max_clusters = max(
        1, min(20, max_clusters_possible if max_clusters_possible > 0 else 1)
    )
    default_n_clusters = max(
        min_clusters, min(5, slider_max_clusters) if slider_max_clusters > 0 else 1
    )

    n_clusters_interactive = st.sidebar.slider(
        "Number of Clusters (K-Means)",
        min_value=min_clusters,
        max_value=slider_max_clusters,  # Make sure slider_max_clusters is at least 1
        value=default_n_clusters,
        step=1,
        key="n_clusters_slider_plotly",
    )

    # This was already checked, but good for robustness
    if base_df is None or base_df.empty:
        st.error(
            "No data available to display clusters."
        )  # Should not be reached if initial check passes
        return

    df_clustered, actual_n_clusters = run_clustering_func(
        base_df, n_clusters_interactive, selected_features_actual
    )

    # Message about excluded planets
    # Count planets that *could* have been clustered (had data for selected features)
    num_eligible_for_clustering = 0
    if selected_features_actual:
        # Create a temporary DataFrame with only the selected features and drop rows where ALL are NaN
        # A more accurate count would be rows where NONE of the selected features are NaN,
        # matching run_clustering's dropna(subset=...)
        eligible_df = base_df.dropna(subset=selected_features_actual)
        num_eligible_for_clustering = len(eligible_df)

    num_actually_clustered = len(df_clustered[df_clustered["cluster"] != -1])

    if selected_features_actual:
        if num_actually_clustered < num_eligible_for_clustering:
            st.sidebar.info(
                f"{num_eligible_for_clustering - num_actually_clustered} planet(s) with data for selected features were not clustered (e.g., became singletons or other K-Means edge cases). "
                f"They are in Cluster ID -1 (or unassigned)."
            )
        total_rows_with_any_nan_in_selected = len(base_df) - num_eligible_for_clustering
        if total_rows_with_any_nan_in_selected > 0:
            st.sidebar.info(
                f"{total_rows_with_any_nan_in_selected} planet(s) were missing data for one or more selected features and excluded from clustering eligibility."
            )

    if (
        actual_n_clusters < n_clusters_interactive
        and n_clusters_interactive > 1  # Only show if user requested more than 1
        and num_actually_clustered > 0  # Only if some clustering happened
    ):
        st.sidebar.info(
            f"Adjusted to {actual_n_clusters} clusters for the {num_actually_clustered} planet(s) included in clustering, likely due to data characteristics or number of valid data points."
        )

    # Ensure hover_text_main exists before trying to use it
    if (
        "hover_text_main" not in df_clustered.columns
        and "pl_name" in df_clustered.columns
    ):  # Basic fallback
        df_clustered["hover_text_main"] = df_clustered["pl_name"]

    df_clustered["hover_text_cluster"] = df_clustered.apply(
        lambda row: f"Cluster ID: {format_value(row.get('cluster'), 'int')}<br>"
        + row.get(
            "hover_text_main",
            f"Planet: {row.get('pl_name', 'N/A')} Details not available",
        ),
        axis=1,
    )

    fig = create_base_figure(
        df_clustered, neighborhood_sphere_initial_visibility=True
    )  # create_base_figure should handle None or empty df_clustered

    if (
        not df_clustered.empty
        and "cluster" in df_clustered.columns
        and "x" in df_clustered.columns
    ):
        # Filter out rows that don't have coordinates for plotting, even if clustered
        plot_df = df_clustered.dropna(subset=["x", "y", "z", "marker_size", "cluster"])

        if not plot_df.empty:
            show_colorbar = (
                plot_df["cluster"].nunique() > 1
            )  # Base colorbar on actual plotted data
            fig.add_trace(
                go.Scatter3d(
                    x=plot_df["x"],
                    y=plot_df["y"],
                    z=plot_df["z"],
                    mode="markers",
                    marker=dict(
                        size=plot_df["marker_size"],
                        color=plot_df["cluster"],
                        colorscale="viridis",
                        opacity=0.9,
                        colorbar=(
                            dict(
                                title="Cluster ID", thickness=15, len=0.6, y=0.5, x=1.05
                            )
                            if show_colorbar
                            else None
                        ),
                    ),
                    text=plot_df["hover_text_cluster"],
                    hoverinfo="text",
                    name="Exoplanets (Clusters)",
                    showlegend=True,
                    visible=True,
                )
            )
        else:
            st.info(
                "No planets with valid coordinates and cluster assignments to display in the 3D map."
            )

    cluster_features_str = (
        ", ".join(selected_features_friendly)
        if selected_features_friendly
        else "N/A (No features selected for clustering)"
    )
    fig.update_layout(
        title_text=f"Exoplanetary Systems - K-Means Clustered by: {cluster_features_str}"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def display_lda_visualization(
    base_df, run_lda_func, feature_options_map, classification_colors
):
    st.header("Linear Discriminant Analysis (LDA) Visualization")
    st.markdown(
        "LDA is a supervised dimensionality reduction technique that projects data to a lower-dimensional "
        "space while maximizing separability between predefined classes. "
        "Here, we use the **'classification_category'** as the target classes. "
        "The plot shows the first two LDA components."
    )

    if base_df is None or base_df.empty:
        st.warning("No exoplanet data available for LDA.")
        return

    selected_lda_features_friendly = [
        "Planet Radius (Earth Radii)",
        "Equilibrium Temperature (K)",
        "Insolation Flux (Earth Flux)",
        "Planet Density (g/cm³)",
        "Stellar Eff. Temp. (K)",
        "Stellar Radius (Solar Radii)",
        "Stellar Mass (Solar Masses)",
        "Stellar Metallicity ([Fe/H] dex)",
    ]

    selected_lda_features_actual = [
        feature_options_map[f] for f in selected_lda_features_friendly
    ]

    if not selected_lda_features_actual:
        st.sidebar.warning("Please select at least one feature for LDA.")
        st.info("Select features in the sidebar to perform LDA analysis.")
        return

    df_with_lda, lda_model, actual_n_components, explained_variance, messages = (
        run_lda_func(
            base_df,
            selected_lda_features_actual,
            target_column_name="classification_category",
            n_components_to_request=2,
        )
    )

    for msg_type, msg_list in messages.items():
        for msg in msg_list:
            if msg_type == "warning":
                st.sidebar.warning(msg)
            elif msg_type == "error":
                st.error(msg)
            elif msg_type == "info":
                st.info(msg)

    if actual_n_components > 0 and "lda_comp_1" in df_with_lda.columns:
        df_plot_lda = df_with_lda.dropna(subset=["lda_comp_1"]).copy()
    else:
        df_plot_lda = pd.DataFrame()

    if lda_model is None or actual_n_components == 0 or df_plot_lda.empty:
        st.info(
            "LDA could not be performed or resulted in no components for the current data/feature selection."
        )
        return

    st.subheader(
        f"LDA Results ({actual_n_components} Component{'s' if actual_n_components > 1 else ''})"
    )

    # --- THIS IS THE CORRECTED PART ---
    # Check if explained_variance is not None and has elements
    if explained_variance is not None and len(explained_variance) > 0:
        # Alternatively, if explained_variance is always a numpy array (even if empty):
        # if explained_variance.size > 0:
        # --- END OF CORRECTION ---
        expl_var_str_list = [
            f"LD{i+1}: {var*100:.2f}%" for i, var in enumerate(explained_variance)
        ]
        st.write(
            f"Explained Variance Ratio by component: {', '.join(expl_var_str_list)}"
        )
        st.write(
            f"Total Explained Variance ({min(actual_n_components, len(explained_variance))} components): {sum(explained_variance)*100:.2f}%"
        )

    def create_lda_hover_text(row):
        base_hover = row.get("hover_text_main", "")
        # Ensure robust splitting of hover_text_main
        parts = base_hover.split("--- Key Parameters ---", 1)
        header_part = parts[0]
        params_part = parts[1] if len(parts) > 1 else "Parameter details not available."

        lda_info_list = []
        if "lda_comp_1" in row and pd.notna(row["lda_comp_1"]):
            lda_info_list.append(f"LDA Comp 1: {row['lda_comp_1']:.3f}")
        if (
            actual_n_components >= 2
            and "lda_comp_2" in row
            and pd.notna(row.get("lda_comp_2"))
        ):
            lda_info_list.append(f"LDA Comp 2: {row['lda_comp_2']:.3f}")

        lda_info_str = "<br>".join(lda_info_list)
        if lda_info_str:
            lda_info_str += "<br>"

        return f"{header_part.replace('Classification:', 'Original Class:')}<br>{lda_info_str}<br>--- Key Parameters ---{params_part}"

    df_plot_lda.loc[:, "hover_text_lda"] = df_plot_lda.apply(
        create_lda_hover_text, axis=1
    )

    if actual_n_components == 1:
        st.write("LDA resulted in 1 component. Displaying as a histogram/density plot.")
        fig_lda = px.histogram(
            df_plot_lda,
            x="lda_comp_1",
            color="classification_category",
            marginal="box",
            color_discrete_map=classification_colors,
            labels={"lda_comp_1": "LDA Component 1"},
            title="Distribution of LDA Component 1 by Class",
            custom_data=["hover_text_lda"],
        )
        fig_lda.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>"
        )  # Ensure hover works
        st.plotly_chart(fig_lda, use_container_width=True)

    elif actual_n_components >= 2:
        st.write(
            "LDA resulted in 2 or more components. Displaying the first two components."
        )
        fig_lda = px.scatter(
            df_plot_lda,
            x="lda_comp_1",
            y="lda_comp_2",
            color="classification_category",
            color_discrete_map=classification_colors,
            symbol="classification_category",
            labels={"lda_comp_1": "LDA Component 1", "lda_comp_2": "LDA Component 2"},
            title="Exoplanets in LDA Space (First 2 Components)",
            custom_data=["hover_text_lda"],  # Ensure hover works
        )
        fig_lda.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>"
        )  # Ensure hover works
        fig_lda.update_layout(legend_title_text="Classification")
        st.plotly_chart(fig_lda, use_container_width=True)

    if (
        hasattr(lda_model, "scalings_")
        and lda_model.scalings_ is not None
        and lda_model.scalings_.size > 0
    ):
        st.subheader("LDA Loadings (Feature Contributions to LDA Components)")
        num_loadings_to_show = min(actual_n_components, lda_model.scalings_.shape[1], 2)
        if num_loadings_to_show > 0:
            loadings_cols = [f"LD{i+1}" for i in range(num_loadings_to_show)]

            # Ensure selected_lda_features_actual matches the number of rows in scalings_
            # This means it should be the features *used* by LDA
            # `lda_model.feature_names_in_` would be ideal if available and consistently set,
            # but selected_lda_features_actual (after scaling/LDA processing) should align with scalings_ rows.

            # If lda_model.scalings_ has fewer rows than selected_lda_features_actual due to feature removal
            # (e.g., zero variance), this could be an issue. For now, assume they align.
            # A safer approach would be to get feature names from the lda_model if possible,
            # or ensure the input features to LDA are exactly those used for scaling.

            if len(selected_lda_features_actual) == lda_model.scalings_.shape[0]:
                loadings_df = pd.DataFrame(
                    lda_model.scalings_[:, :num_loadings_to_show],
                    index=selected_lda_features_actual,
                    columns=loadings_cols,
                )
                st.write(
                    "Loadings indicate how much each original feature contributes to the LDA components."
                )
                st.dataframe(
                    loadings_df.style.format("{:.3f}").background_gradient(
                        cmap="viridis", axis=0
                    )
                )
            else:
                st.warning(
                    f"Could not display LDA loadings: Mismatch between number of selected features ({len(selected_lda_features_actual)}) and features in LDA model scalings ({lda_model.scalings_.shape[0]})."
                )
        else:
            st.write(
                "No LDA loadings to display (e.g., no components or scalings not available)."
            )
