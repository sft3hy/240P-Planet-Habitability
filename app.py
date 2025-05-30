# app.py
import streamlit as st
from data_processing import (
    load_and_prepare_data,
    run_clustering,
)  # Import run_clustering
from plotting import display_cluster_visualization, display_habitability_visualization

# Load data once
df_processed = load_and_prepare_data()  # Using default file path

if df_processed is not None:
    # For Cluster Analysis Page/Section
    # Pass the actual run_clustering function from data_processing
    display_cluster_visualization(df_processed, run_clustering)

    # For Habitability Page/Section
    display_habitability_visualization(df_processed)
else:
    st.error("Data could not be loaded. Visualizations cannot be displayed.")
