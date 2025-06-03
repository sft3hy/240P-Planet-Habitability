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
    CLASSIFICATION_COLORS,
    HOVER_FONT_COLOR,
    HOVER_BG_COLORS,
)

# format_value is implicitly available if called from data_processing, but explicit import is safer
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
    # Add Sun
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
    # Add Local Neighborhood Sphere
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

    # Dynamic Camera
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
        margin=dict(l=0, r=0, b=0, t=40),  # Keep title margin small
    )
    return fig


def display_exoplanet_classification_map(df_input):
    st.header("Exoplanet Classification Map")
    st.markdown(
        "3D visualization of exoplanetary systems, colored by their habitability classification based on the paper's criteria. Hover over planets for detailed information."
    )

    if df_input is None or df_input.empty:
        st.warning("No exoplanet data available to display.")
        return

    df = df_input.copy()
    fig = create_base_figure(df, neighborhood_sphere_initial_visibility="legendonly")

    # Add traces for each classification category for legend and distinct coloring
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
                        color=color,  # Use the direct color string
                        opacity=0.9,
                    ),
                    text=category_df["hover_text_main"],
                    hoverinfo="text",
                    name=category,  # For legend
                    hoverlabel=dict(
                        bgcolor=HOVER_BG_COLORS.get(
                            category, "grey"
                        ),  # Fallback bgcolor
                        font=dict(color=HOVER_FONT_COLOR),
                        bordercolor="rgba(0,0,0,0.6)",
                        namelength=-1,  # Show full hover text
                    ),
                )
            )
        else:  # Add an empty trace if no planets in this category, for legend completeness
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

    st.subheader("Summary of Key Paper Insights")
    good_star_poor_planet_percentage = classification_percentages.get(
        "Good Star, Poor Planet", 0
    )
    st.markdown(
        f"**Detection Bias:** Approximately **{good_star_poor_planet_percentage:.1f}%** of classified exoplanets fall into the 'Good Star, Poor Planet' category, indicating their host stars are suitable but the planets are not. This highlights observational biases towards easier-to-detect, often less habitable, planets."
    )

    st.markdown(
        "**Rarity of Earth-like Conditions:** Earth's statistical position (e.g., ~70th percentile for unusualness in the paper) suggests Earth-like conditions are uncommon but achievable. Only a small fraction (e.g., 0.6% in the paper, adjust based on current data) are 'Excellent Candidates', emphasizing their rarity while acknowledging that detection bias inflates this perceived scarcity."
    )


def display_cluster_visualization(base_df, run_clustering_func):
    st.header("K-Means Cluster Analysis")
    st.markdown(
        "Explore exoplanet clusters based on selected physical parameters. This is an unsupervised learning approach to find natural groupings in the data."
    )

    feature_options_map = {
        "Planet Radius (ER)": "pl_rade",
        "Equilibrium Temperature (K)": "pl_eqt",
        "Insolation (Earth Flux)": "pl_insol",
        "Planet Density (g/cm³)": "pl_dens",
        "Stellar Eff. Temp. (K)": "st_teff",
        "Stellar Radius (SR)": "st_rad",
        "Stellar Mass (SM)": "st_mass",
        "Stellar Metallicity (dex)": "st_met",
    }
    default_selection_friendly = ["Planet Radius (ER)", "Equilibrium Temperature (K)"]
    valid_default_selection = [
        f for f in default_selection_friendly if f in feature_options_map
    ]

    selected_features_friendly = st.sidebar.multiselect(
        "Select features for K-Means clustering:",
        options=list(feature_options_map.keys()),
        default=valid_default_selection,
        key="cluster_features_multiselect",
    )
    selected_features_actual = [
        feature_options_map[f] for f in selected_features_friendly
    ]
    if not selected_features_actual:
        st.sidebar.warning("Please select at least one feature for clustering.")

    min_clusters, max_clusters_possible = 1, (
        len(base_df) if base_df is not None and not base_df.empty else 1
    )
    slider_max_clusters = max(
        1, min(20, max_clusters_possible if max_clusters_possible > 0 else 1)
    )
    default_n_clusters = max(
        min_clusters, min(5, slider_max_clusters) if slider_max_clusters > 0 else 1
    )  # Default to 5 or less

    n_clusters_interactive = st.sidebar.slider(
        "Number of Clusters (K-Means)",
        min_value=min_clusters,
        max_value=slider_max_clusters,
        value=default_n_clusters,
        step=1,
        key="n_clusters_slider_plotly",
    )

    if base_df is None:
        st.error("No data available to display clusters.")
        return

    df_clustered, actual_n_clusters = run_clustering_func(
        base_df, n_clusters_interactive, selected_features_actual
    )

    num_actually_clustered = len(df_clustered[df_clustered["cluster"] != -1])
    if selected_features_actual and num_actually_clustered < len(df_clustered):
        st.sidebar.info(
            f"{len(df_clustered) - num_actually_clustered} planet(s) excluded from clustering (e.g., missing data for selected features) and are in Cluster ID -1."
        )
    if (
        actual_n_clusters < n_clusters_interactive
        and len(base_df) > 1
        and n_clusters_interactive > 1
        and num_actually_clustered > 0
    ):
        st.sidebar.info(
            f"Adjusted to {actual_n_clusters} clusters for the {num_actually_clustered} planet(s) included in clustering due to data characteristics."
        )

    df_clustered["hover_text_cluster"] = df_clustered.apply(
        lambda row: f"Cluster ID: {format_value(row.get('cluster'), 'int')}<br>"
        + row["hover_text_main"],
        axis=1,
    )

    fig = create_base_figure(df_clustered, neighborhood_sphere_initial_visibility=True)
    if not df_clustered.empty and "cluster" in df_clustered.columns:
        show_colorbar = df_clustered["cluster"].nunique() > 1
        fig.add_trace(
            go.Scatter3d(
                x=df_clustered["x"],
                y=df_clustered["y"],
                z=df_clustered["z"],
                mode="markers",
                marker=dict(
                    size=df_clustered["marker_size"],  # Use the unified marker size
                    color=df_clustered["cluster"],
                    colorscale="viridis",  # Viridis handles -1 well
                    opacity=0.9,
                    colorbar=(
                        dict(title="Cluster ID", thickness=15, len=0.6, y=0.5, x=1.05)
                        if show_colorbar
                        else None
                    ),
                ),
                text=df_clustered["hover_text_cluster"],
                hoverinfo="text",
                name="Exoplanets (Clusters)",
                showlegend=True,  # Keep one legend entry for "Exoplanets (Clusters)"
                visible=True,
            )
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
