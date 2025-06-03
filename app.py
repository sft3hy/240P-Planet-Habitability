# app.py
import streamlit as st

from data_processing import load_and_prepare_data, run_clustering
from plotting import (
    display_exoplanet_classification_map,
    display_habitability_analysis_dashboard,
    display_cluster_visualization,
)
from constants import KEY_PARAMETERS_MAP, HABITABILITY_THRESHOLDS, CLASSIFICATION_COLORS


def main():
    st.set_page_config(
        page_title="Exoplanet Habitability Explorer", layout="wide", page_icon="🌌"
    )
    st.sidebar.title("Exoplanet Explorer")
    st.sidebar.markdown("An interactive tool for exploring exoplanet habitability")

    base_df = load_and_prepare_data()

    if base_df is None or base_df.empty:
        st.sidebar.error("Data loading failed. Please check data file and try again.")
        st.error(
            "Critical error: Exoplanet data could not be loaded. The application cannot proceed."
        )
        return

    page_options = [
        "Exoplanet Classification Map",
        "Habitability Analysis & Insights",
        "K-Means Cluster Analysis",
    ]
    selected_page = st.sidebar.radio(
        "Select Analysis View", page_options, key="main_page_selection"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Data Source: NASA Exoplanet Archive (accessed May 20, 2025).")

    if selected_page == "Exoplanet Classification Map":
        display_exoplanet_classification_map(base_df)
    elif selected_page == "Habitability Analysis & Insights":
        display_habitability_analysis_dashboard(
            base_df, KEY_PARAMETERS_MAP, HABITABILITY_THRESHOLDS, CLASSIFICATION_COLORS
        )
    elif selected_page == "K-Means Cluster Analysis":
        display_cluster_visualization(base_df, run_clustering)


if __name__ == "__main__":
    main()
