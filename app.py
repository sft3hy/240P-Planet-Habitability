# app.py
import streamlit as st

from data_processing import (
    load_and_prepare_data,
    run_clustering,
    run_lda_analysis,
)
from plotting import (
    display_exoplanet_classification_map,
    display_habitability_analysis_dashboard,
    display_cluster_visualization,
    display_lda_visualization,
    display_pca_analysis_page,
)
from constants import (
    KEY_PARAMETERS_MAP,
    HABITABILITY_THRESHOLDS,
    CLASSIFICATION_COLORS,
)


def main():
    st.set_page_config(
        page_title="Exoplanet Habitability Explorer", layout="wide", page_icon="🌌"
    )
    st.sidebar.title("Exoplanet Explorer")
    st.sidebar.markdown("An interactive tool for exploring exoplanet habitability")

    # Main data loading for most pages
    base_df = load_and_prepare_data()

    if base_df is None or base_df.empty:
        st.sidebar.error(
            "Data loading failed for main application. Please check data file and try again."
        )
        st.error(
            "Critical error: Main exoplanet data could not be loaded. Some application features may not work."
        )

    feature_options_map_ui = {}
    if KEY_PARAMETERS_MAP:
        for col_name, details in KEY_PARAMETERS_MAP.items():
            if "name" in details and "unit" in details:
                feature_options_map_ui[f"{details['name']} ({details['unit']})"] = (
                    col_name
                )
            elif "name" in details:
                feature_options_map_ui[f"{details['name']}"] = col_name

    if not feature_options_map_ui and (base_df is not None and not base_df.empty):
        st.sidebar.warning(
            "Could not generate feature selection options from KEY_PARAMETERS_MAP."
        )
        feature_options_map_ui = {
            "Planet Radius (ER)": "pl_rade",
            "Equilibrium Temperature (K)": "pl_eqt",
            "Insolation (Earth Flux)": "pl_insol",
            "Planet Density (g/cm³)": "pl_dens",
            "Stellar Eff. Temp. (K)": "st_teff",
            "Stellar Radius (SR)": "st_rad",
            "Stellar Mass (SM)": "st_mass",
            "Stellar Metallicity (dex)": "st_met",
        }

    page_options = [
        "Exoplanet Classification Map",
        "Habitability Analysis & Insights",
        "K-Means Cluster Analysis",
        "LDA Visualization",
        "PCA Analysis",
    ]
    selected_page = st.sidebar.radio(
        "Select Analysis View", page_options, key="main_page_selection"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Data Source: NASA Exoplanet Archive (accessed May 20, 2025).")

    if selected_page == "Exoplanet Classification Map":
        if base_df is not None and not base_df.empty:
            display_exoplanet_classification_map(base_df)
        else:
            st.error("Data not available for Exoplanet Classification Map.")
    elif selected_page == "Habitability Analysis & Insights":
        if base_df is not None and not base_df.empty:
            display_habitability_analysis_dashboard(
                base_df,
                KEY_PARAMETERS_MAP,
                HABITABILITY_THRESHOLDS,
                CLASSIFICATION_COLORS,
            )
        else:
            st.error("Data not available for Habitability Analysis & Insights.")
    elif selected_page == "K-Means Cluster Analysis":
        if base_df is not None and not base_df.empty and feature_options_map_ui:
            display_cluster_visualization(
                base_df, run_clustering, feature_options_map_ui
            )
        else:
            st.error(
                "Data or feature options not available for K-Means Cluster Analysis."
            )
    elif selected_page == "LDA Visualization":
        if base_df is not None and not base_df.empty and feature_options_map_ui:
            display_lda_visualization(
                base_df, run_lda_analysis, feature_options_map_ui, CLASSIFICATION_COLORS
            )
        else:
            st.error("Data or feature options not available for LDA Visualization.")
    elif selected_page == "PCA Analysis":
        display_pca_analysis_page()


if __name__ == "__main__":
    main()
