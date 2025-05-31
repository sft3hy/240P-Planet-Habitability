# app.py
import streamlit as st

# Import functions from other modules
from data_processing import load_and_prepare_data, run_clustering
from plotting import display_cluster_visualization, display_habitability_visualization


def main():
    st.set_page_config(page_title="Exoplanet Explorer", layout="wide")
    st.sidebar.title("Exoplanet Explorer")

    # Load and prepare data once
    base_df = load_and_prepare_data()  # Default file path is used

    if base_df is None or base_df.empty:
        # Error/warning messages are handled within load_and_prepare_data
        st.sidebar.warning("Halting app due to data loading issues.")
        return

    page_options = ["Cluster Analysis", "Habitability Status"]
    selected_page = st.sidebar.radio(
        "Select Visualization", page_options, key="main_page_selection"
    )

    if selected_page == "Cluster Analysis":
        display_cluster_visualization(base_df, run_clustering)
    elif selected_page == "Habitability Status":
        display_habitability_visualization(base_df)


if __name__ == "__main__":
    main()
