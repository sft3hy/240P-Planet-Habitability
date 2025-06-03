# data_processing.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import warnings

from constants import (
    PARSEC_TO_LY,
    HABITABILITY_THRESHOLDS,
    KEY_PARAMETERS_MAP,
    DEFAULT_PLANET_MARKER_SIZE,
    EXCELLENT_CANDIDATE_MARKER_SIZE,
    EARTH_MARKER_SIZE,
)

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.utils.extmath"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="KMeans is known to have a memory leak on Windows with MKL",
)


def spherical_to_cartesian(ra, dec, dist_ly):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = dist_ly * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist_ly * np.cos(dec_rad) * np.sin(ra_rad)
    z = dist_ly * np.sin(dec_rad)
    return x, y, z


def format_value(value, precision_type):
    if pd.isna(value):
        return "N/A"
    if precision_type == "int":
        try:
            return f"{int(value)}"
        except (ValueError, TypeError):
            return "N/A"
    if isinstance(precision_type, int):
        try:
            return f"{float(value):.{precision_type}f}"
        except (ValueError, TypeError):
            return "N/A"
    return str(value)


def classify_exoplanet(row, thresholds, param_map):
    p_good = 1
    s_good = 1

    all_params_present = all(pd.notna(row.get(key)) for key in param_map.keys())
    if not all_params_present:
        return "Unclassified (Missing Data)"

    for param_key, details in param_map.items():
        value = row.get(param_key)  # Already checked for NaN above overall
        param_range = thresholds[param_key]
        if not (param_range["min"] <= value <= param_range["max"]):
            if details["type"] == "planetary":
                p_good = 0
            elif details["type"] == "stellar":
                s_good = 0

    if p_good == 1 and s_good == 1:
        return "Excellent Candidate"
    elif p_good == 1 and s_good == 0:
        return "Good Planet, Poor Star"
    elif p_good == 0 and s_good == 1:
        return "Good Star, Poor Planet"
    else:  # p_good == 0 and s_good == 0
        return "Poor Candidate"


@st.cache_data
def load_and_prepare_data(file_path="data/Planetary-Systems-May-20-2025_clean.csv"):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(
            f"Error: Data file '{file_path}' not found. Please make sure it exists."
        )
        return None

    if df.empty:
        st.warning("Loaded DataFrame is empty. No data to plot.")
        return None

    coord_cols = ["ra", "dec", "sy_dist"]
    if not all(col in df.columns for col in coord_cols):
        st.error(f"Missing one or more coordinate columns {coord_cols} in the CSV.")
        return None

    df = df.dropna(subset=coord_cols).copy()
    if df.empty:
        st.warning("DataFrame is empty after dropping NaNs in coordinate columns.")
        return None

    df["sy_dist_ly"] = df["sy_dist"] * PARSEC_TO_LY

    earth_data = {
        "pl_name": "Earth",
        "hostname": "Sol",  # Simplified name
        "sy_dist": 0.0,
        "sy_dist_ly": 0.0,
        "ra": 0.0,
        "dec": 0.0,
        "glon": np.nan,
        "glat": np.nan,
        "pl_rade": 1.0,
        "pl_eqt": 288.0,
        "pl_insol": 1.0,
        "pl_dens": 5.51,
        "st_teff": 5778.0,
        "st_rad": 1.0,
        "st_mass": 1.0,
        "st_met": 0.0,
    }
    for key in KEY_PARAMETERS_MAP.keys():
        if key not in earth_data:
            earth_data[key] = np.nan

    # Use a more robust way to check for Earth, allowing for variations like "Earth (Sol System)"
    earth_mask = df["pl_name"].astype(str).str.contains("Earth", case=False) & df[
        "hostname"
    ].astype(str).str.contains("Sol", case=False, na=False)

    if not earth_mask.any():
        earth_df_row = pd.DataFrame([earth_data])
        df = pd.concat([df, earth_df_row], ignore_index=True)
    else:  # Update existing Earth entry
        earth_idx = df[earth_mask].index[0]  # Get the first match
        for col, val in earth_data.items():
            df.loc[earth_idx, col] = val
        # Ensure pl_name and hostname are standardized for our Earth entry
        df.loc[earth_idx, "pl_name"] = "Earth"
        df.loc[earth_idx, "hostname"] = "Sol"

    df["classification_category"] = df.apply(
        classify_exoplanet, args=(HABITABILITY_THRESHOLDS, KEY_PARAMETERS_MAP), axis=1
    )

    # Ensure Earth is correctly classified
    earth_final_mask = (df["pl_name"] == "Earth") & (df["hostname"] == "Sol")
    if earth_final_mask.any():
        earth_idx_final = df[earth_final_mask].index[0]
        earth_row_values = df.loc[earth_idx_final]
        df.loc[earth_idx_final, "classification_category"] = classify_exoplanet(
            earth_row_values, HABITABILITY_THRESHOLDS, KEY_PARAMETERS_MAP
        )

    df["x"], df["y"], df["z"] = spherical_to_cartesian(
        df["ra"], df["dec"], df["sy_dist_ly"]
    )

    df["hover_text_main"] = df.apply(
        lambda row: f"""<b>{row.get('pl_name', 'N/A')}</b> ({row.get('hostname', 'N/A')})<br>
Classification: <b>{row.get('classification_category', 'N/A')}</b><br>
Distance: {format_value(row.get('sy_dist_ly'), 2)} ly<br>
--- Key Parameters ---
Radius: {format_value(row.get('pl_rade'), 2)} ER | Density: {format_value(row.get('pl_dens'), 2)} g/cm³<br>
Eq. Temp: {format_value(row.get('pl_eqt'), 1)} K | Insol: {format_value(row.get('pl_insol'), 2)} EF<br>
Stellar Teff: {format_value(row.get('st_teff'), 0)} K | Stellar Rad: {format_value(row.get('st_rad'), 2)} SR<br>
Stellar Mass: {format_value(row.get('st_mass'), 2)} SM | Stellar Met: {format_value(row.get('st_met'), 2)} dex
""".strip(),
        axis=1,
    )

    def get_marker_size(row):
        if row["pl_name"] == "Earth" and row["hostname"] == "Sol":
            return EARTH_MARKER_SIZE
        if row["classification_category"] == "Excellent Candidate":
            return EXCELLENT_CANDIDATE_MARKER_SIZE
        return DEFAULT_PLANET_MARKER_SIZE

    df["marker_size"] = df.apply(get_marker_size, axis=1)

    return df


@st.cache_data
def run_clustering(_df_input, n_clusters_requested, cluster_features_list):
    df = _df_input.copy()
    df["cluster"] = -1  # Signifies "not clustered" or "unclusterable"

    if not cluster_features_list:
        return df, 1

    missing_cols = [col for col in cluster_features_list if col not in df.columns]
    if missing_cols:
        return df, 1  # Silently return if features are missing

    features_for_clustering_df = df[cluster_features_list].copy()
    features_for_clustering_df.dropna(subset=cluster_features_list, inplace=True)
    valid_indices_for_clustering = features_for_clustering_df.index

    if features_for_clustering_df.empty:
        return df, 1

    features_for_clustering_values = features_for_clustering_df.values
    if len(features_for_clustering_values) < 2:
        if len(features_for_clustering_values) == 1:
            df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    if (
        pd.DataFrame(features_for_clustering_values, columns=cluster_features_list)
        .nunique()
        .max()
        == 1
    ):
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    actual_n_clusters = max(
        1, min(n_clusters_requested, len(features_for_clustering_values))
    )
    if actual_n_clusters == 1:
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features_for_clustering_values)
    except ValueError:  # Should be caught by nunique, but safeguard
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    if np.any(np.isnan(scaled_features)) or np.any(np.isinf(scaled_features)):
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    try:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(scaled_features)
        df.loc[valid_indices_for_clustering, "cluster"] = cluster_labels
    except Exception:  # Fallback if KMeans errors on prepared data
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        actual_n_clusters = 1
    return df, actual_n_clusters
