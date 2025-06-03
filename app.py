# app.py
import streamlit as st

from data_processing import (
    load_and_prepare_data,
    run_clustering,
    run_lda_analysis,
)  # Added run_lda_analysis
from plotting import (
    display_exoplanet_classification_map,
    display_habitability_analysis_dashboard,
    display_cluster_visualization,
    display_lda_visualization,  # Added display_lda_visualization
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

    # Create feature_options_map_ui from KEY_PARAMETERS_MAP
    # This map provides user-friendly names for features (e.g., "Planet Radius (ER)")
    # and maps them to the actual DataFrame column names (e.g., "pl_rade").
    feature_options_map_ui = {}
    if KEY_PARAMETERS_MAP:
        for col_name, details in KEY_PARAMETERS_MAP.items():
            if "name" in details and "unit" in details:
                feature_options_map_ui[f"{details['name']} ({details['unit']})"] = (
                    col_name
                )
            elif "name" in details:  # Fallback if unit is missing
                feature_options_map_ui[f"{details['name']}"] = col_name

    if not feature_options_map_ui:
        st.sidebar.warning(
            "Could not generate feature selection options from KEY_PARAMETERS_MAP."
        )
        # Provide a default or empty map to prevent errors downstream if KEY_PARAMETERS_MAP is empty/malformed
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
        "LDA Visualization",  # Added LDA option
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
        # Assuming display_cluster_visualization is updated to accept feature_options_map
        display_cluster_visualization(base_df, run_clustering, feature_options_map_ui)
    elif selected_page == "LDA Visualization":  # Added LDA page
        display_lda_visualization(
            base_df,
            run_lda_analysis,  # Pass the LDA processing function
            feature_options_map_ui,  # Pass the UI map for feature selection
            CLASSIFICATION_COLORS,  # Pass color mapping for classes
        )


if __name__ == "__main__":
    main()
