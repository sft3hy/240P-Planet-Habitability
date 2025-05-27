# plotting.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd  # For type hinting if necessary

# Import constants
from constants import (
    APP_SUN_MARKER_SIZE,
    LOCAL_NEIGHBORHOOD_OPACITY,
    SOLAR_SYSTEM_OPACITY,
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
from data_processing import format_value


def create_sphere_surface(radius, color, opacity=0.1, name="", resolution=20):
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
        surfacecolor=np.full_like(x_sphere, 0.5),  # Dummy array for uniform color
        cmin=0,
        cmax=1,  # Lock colorscale for uniform color
        showscale=False,
        opacity=opacity,
        name=name,
        hoverinfo="skip",  # Skip hover for surface itself
    )


def create_base_figure(df):
    fig = go.Figure()

    # Add Sun Marker (will appear in legend)
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=APP_SUN_MARKER_SIZE, color="yellow", opacity=1.0),
            name="Our Sun (Sol)",
            hoverinfo="name",
        )
    )

    # Add Solar System Boundary Sphere (will not appear in legend due to hoverinfo='skip' in create_sphere_surface)
    SOLAR_SYSTEM_EXTENT_AU = 100
    solar_radius_actual_ly = SOLAR_SYSTEM_EXTENT_AU * 0.000015813  # AU to Ly
    fig.add_trace(
        create_sphere_surface(
            radius=solar_radius_actual_ly,
            color="gold",
            opacity=SOLAR_SYSTEM_OPACITY,
            name=f"Solar System Extent (~{SOLAR_SYSTEM_EXTENT_AU} AU)",
        )
    )

    # Add Local Neighborhood Sphere (will not appear in legend)
    neighborhood_radius_ly = 1000
    fig.add_trace(
        create_sphere_surface(
            radius=neighborhood_radius_ly,
            color="deepskyblue",
            opacity=LOCAL_NEIGHBORHOOD_OPACITY,
            name=f"Local Neighborhood (~{neighborhood_radius_ly} ly)",
        )
    )

    # Camera position logic
    camera_eye_x, camera_eye_y, camera_eye_z = (
        CAMERA_DEFAULT_EYE_X,
        CAMERA_DEFAULT_EYE_Y,
        CAMERA_DEFAULT_EYE_Z,
    )
    if df is not None and not df.empty and "sy_dist_ly" in df.columns:
        max_dist = df["sy_dist_ly"].max()
        if pd.notna(max_dist) and max_dist > 0:
            # Clip effective_distance to control zoom range more predictably
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
        legend=LEGEND_STYLE,  # Use the defined legend style
        margin=dict(l=0, r=0, b=0, t=40),  # Margin for title
    )
    return fig


def display_cluster_visualization(base_df, run_clustering_func):
    st.header("Exoplanet Visualization: Cluster Analysis")

    min_clusters = 1
    # Ensure base_df is not None and not empty before accessing its length
    max_clusters_possible = (
        len(base_df) if base_df is not None and not base_df.empty else 1
    )
    default_n_clusters = (
        min(8, max_clusters_possible) if max_clusters_possible > 0 else 1
    )

    n_clusters_interactive = st.sidebar.slider(
        "Number of Clusters (KMeans)",
        min_value=1,
        # Ensure max_value is always >= min_value
        max_value=20,
        value=default_n_clusters,
        step=1,
        key="n_clusters_slider_plotly",  # Unique key for the widget
    )

    df_clustered, actual_n_clusters = run_clustering_func(
        base_df, n_clusters_interactive
    )

    if (
        actual_n_clusters < n_clusters_interactive
        and len(base_df) > 1
        and n_clusters_interactive > 1
    ):
        st.sidebar.info(
            f"Adjusted to {actual_n_clusters} clusters due to data characteristics."
        )

    # Prepare hover text specifically for this page
    df_clustered["hover_text_cluster"] = df_clustered.apply(
        lambda row: f"Cluster ID: {format_value(row.get('cluster'), 'int')}<br>"
        + row["base_hover_text"],
        axis=1,
    )

    fig = create_base_figure(df_clustered)  # Pass df_clustered for camera setup

    if (
        not df_clustered.empty
        and "cluster" in df_clustered.columns
        and not df_clustered["cluster"].empty
    ):
        fig.add_trace(
            go.Scatter3d(
                x=df_clustered["x"],
                y=df_clustered["y"],
                z=df_clustered["z"],
                mode="markers",
                marker=dict(
                    size=df_clustered["app_marker_size"],
                    color=df_clustered["cluster"],
                    colorscale="viridis",
                    opacity=0.9,
                    # Adjust colorbar position if needed, or let it be default
                    colorbar=(
                        dict(title="Cluster ID", thickness=15, len=0.6, y=0.5, x=1.05)
                        if actual_n_clusters > 1
                        else None
                    ),
                ),
                text=df_clustered["hover_text_cluster"],
                hoverinfo="text",
                name="Exoplanets (Clusters)",  # Legend entry for planet points
            )
        )

    fig.update_layout(title_text="Exoplanetary Systems - Colored by Spatial Cluster")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def display_habitability_visualization(base_df):
    st.header("Exoplanet Visualization: Habitability Status")

    df_hab = base_df.copy()  # Work with a copy for this specific visualization

    # Define color based on habitability for this plot
    df_hab["habitability_color"] = df_hab["is_habitable"].apply(
        lambda hab: HABITABLE_COLOR_STR if hab else NON_HABITABLE_COLOR_STR
    )

    fig = create_base_figure(df_hab)  # Pass df_hab for camera setup

    # Add planet markers
    if not df_hab.empty:
        fig.add_trace(
            go.Scatter3d(
                x=df_hab["x"],
                y=df_hab["y"],
                z=df_hab["z"],
                mode="markers",
                marker=dict(
                    size=df_hab["app_marker_size"],
                    color=df_hab[
                        "habitability_color"
                    ],  # Use the pre-defined color array
                    opacity=0.9,
                ),
                text=df_hab[
                    "base_hover_text"
                ],  # Use base hover text from data_processing
                hoverinfo="text",
                name="Exoplanets",  # Generic name; color shows habitability
            )
        )

        # Add dummy traces specifically for the legend on this page
        # These will appear in the legend defined by LEGEND_STYLE
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(color=HABITABLE_COLOR_STR, size=10),
                name="Habitable Planet",
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
            )
        )

    fig.update_layout(
        title_text="Exoplanetary Systems - Colored by Habitability Criteria"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
