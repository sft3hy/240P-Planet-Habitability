# plotting.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Import constants
from constants import (
    APP_SUN_MARKER_SIZE,
    LOCAL_NEIGHBORHOOD_OPACITY,
    HABITABLE_COLOR_STR,
    NON_HABITABLE_COLOR_STR,
    LEGEND_STYLE,
    CAMERA_DEFAULT_EYE_X,
    CAMERA_DEFAULT_EYE_Y,
    CAMERA_DEFAULT_EYE_Z,
    CAMERA_EYE_FACTOR,
    CAMERA_Z_FACTOR,
    CAMERA_EFFECTIVE_DISTANCE_MIN,
    CAMERA_EFFECTIVE_DISTANCE_MAX,
)

# Import helper functions from data_processing
from data_processing import (
    format_value,
)  # Assuming run_clustering is imported where needed (main app)


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
            marker=dict(size=APP_SUN_MARKER_SIZE, color="yellow", opacity=1.0),
            name="Our Sun (Sol)",
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


def display_cluster_visualization(base_df, run_clustering_func):
    st.header("Exoplanet Visualization: Cluster Analysis")

    # --- UI for selecting features to cluster on ---
    feature_options_map = {
        "Planet Radius (ER)": "pl_rade",
        "Equilibrium Temperature (K)": "pl_eqt",
        "Insolation (Earth Flux)": "pl_insol",
        "Planet Density (g/cm³)": "pl_dens",
    }
    # Default selection: Planet Radius and Equilibrium Temperature
    default_selection_friendly = ["Planet Radius (ER)", "Equilibrium Temperature (K)"]

    # Ensure default selections are valid options present in the map
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
        st.sidebar.warning(
            "Please select at least one feature for meaningful clustering."
        )
        # Clustering will proceed, and run_clustering will assign cluster -1 to all points.

    # --- Slider for number of clusters ---
    min_clusters = 1
    max_clusters_possible = (
        len(base_df) if base_df is not None and not base_df.empty else 1
    )
    # Cap max clusters for slider at 20 or dataset size, whichever is smaller
    slider_max_clusters = max(
        1, min(20, max_clusters_possible if max_clusters_possible > 0 else 1)
    )

    default_n_clusters = min(8, slider_max_clusters) if slider_max_clusters > 0 else 1
    # Ensure default_n_clusters is not less than min_clusters
    default_n_clusters = max(min_clusters, default_n_clusters)

    n_clusters_interactive = st.sidebar.slider(
        "Number of Clusters (KMeans)",
        min_value=min_clusters,
        max_value=slider_max_clusters,
        value=default_n_clusters,
        step=1,
        key="n_clusters_slider_plotly",
        # disabled=not selected_features_actual # Optionally disable if no features selected
    )

    # --- Run clustering ---
    # Ensure base_df is not None before passing
    if base_df is None:
        st.error("No data available to display clusters.")
        return

    df_clustered, actual_n_clusters = run_clustering_func(
        base_df, n_clusters_interactive, selected_features_actual
    )

    num_actually_clustered = len(df_clustered[df_clustered["cluster"] != -1])
    if selected_features_actual and num_actually_clustered < len(df_clustered):
        num_excluded = len(df_clustered) - num_actually_clustered
        st.sidebar.info(
            f"{num_excluded} planet(s) could not be included in clustering "
            f"(e.g., missing data for selected features) and are in Cluster ID -1."
        )

    if (
        actual_n_clusters < n_clusters_interactive
        and len(base_df) > 1
        and n_clusters_interactive > 1
        and num_actually_clustered > 0
    ):  # Only show if clustering happened
        st.sidebar.info(
            f"Adjusted to {actual_n_clusters} clusters for the {num_actually_clustered} planet(s) "
            f"included in clustering due to data characteristics."
        )

    df_clustered["hover_text_cluster"] = df_clustered.apply(
        lambda row: f"Cluster ID: {format_value(row.get('cluster'), 'int')}<br>"
        + row["base_hover_text"],
        axis=1,
    )

    fig = create_base_figure(df_clustered, neighborhood_sphere_initial_visibility=True)

    if not df_clustered.empty and "cluster" in df_clustered.columns:
        # Check if there's more than one unique cluster ID (excluding -1 if we want to hide colorbar for only -1s)
        # For colorbar: show if actual_n_clusters > 1 (for clusters 0 to N-1)
        # OR if -1 is present along with other clusters.
        # Essentially, if there's more than one type of cluster value in the 'cluster' column.
        show_colorbar = df_clustered["cluster"].nunique() > 1

        fig.add_trace(
            go.Scatter3d(
                x=df_clustered["x"],
                y=df_clustered["y"],
                z=df_clustered["z"],
                mode="markers",
                marker=dict(
                    size=df_clustered["app_marker_size"],
                    color=df_clustered["cluster"],
                    colorscale="viridis",  # Viridis handles negative values well
                    opacity=0.9,
                    colorbar=(
                        dict(title="Cluster ID", thickness=15, len=0.6, y=0.5, x=1.05)
                        if show_colorbar
                        else None
                    ),
                    # To make -1 distinct, you could set cmin/cmax if you know the range of positive clusters
                    # e.g., cmin=0, cmax=max(0, df_clustered['cluster'].max())
                    # This would push -1 to one end of colorscale. Default behavior is usually fine.
                ),
                text=df_clustered["hover_text_cluster"],
                hoverinfo="text",
                name="Exoplanets (Clusters)",
                showlegend=True,
                visible=True,
            )
        )

    cluster_features_str = (
        ", ".join(selected_features_friendly)
        if selected_features_friendly
        else "N/A (No features selected)"
    )
    fig.update_layout(
        title_text=f"Exoplanetary Systems - Clustered by: {cluster_features_str}"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def display_habitability_visualization(base_df):
    st.header("Exoplanet Visualization: Habitability Status")
    if base_df is None:  # Check if base_df is loaded
        st.error("No data available to display habitability.")
        return

    df_hab = base_df.copy()

    df_hab["habitability_marker_color"] = df_hab["is_habitable"].apply(
        lambda hab: HABITABLE_COLOR_STR if hab else NON_HABITABLE_COLOR_STR
    )

    HOVER_BG_HABITABLE = "navy"
    HOVER_BG_NON_HABITABLE = "darkred"
    HOVER_FONT_COLOR = "white"

    df_hab["hover_bgcolor"] = df_hab["is_habitable"].apply(
        lambda hab: HOVER_BG_HABITABLE if hab else HOVER_BG_NON_HABITABLE
    )
    df_hab["hover_font_color"] = HOVER_FONT_COLOR

    fig = create_base_figure(
        df_hab, neighborhood_sphere_initial_visibility="legendonly"
    )

    if not df_hab.empty:
        fig.add_trace(
            go.Scatter3d(
                x=df_hab["x"],
                y=df_hab["y"],
                z=df_hab["z"],
                mode="markers",
                marker=dict(
                    size=df_hab["app_marker_size"],
                    color=df_hab["habitability_marker_color"],
                    opacity=0.9,
                ),
                text=df_hab["base_hover_text"],
                hoverinfo="text",
                name="Exoplanets",
                showlegend=False,
                visible=True,
                hoverlabel=dict(
                    bgcolor=df_hab["hover_bgcolor"],
                    font=dict(color=df_hab["hover_font_color"]),
                    bordercolor="rgba(0,0,0,0.6)",
                    namelength=-1,
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(color=HABITABLE_COLOR_STR, size=10),
                name="Habitable Planet",
                showlegend=True,
                visible=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(color=NON_HABITABLE_COLOR_STR, size=10),
                name="Non-Habitable Planet",
                showlegend=True,
                visible=True,
            )
        )

    fig.update_layout(
        title_text="Exoplanetary Systems - Colored by Habitability Criteria"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
