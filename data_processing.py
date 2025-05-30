# data_processing.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st  # Keep for st.cache_data
import warnings

# Import constants
from constants import (
    PARSEC_TO_LY,
    EQT_MIN,
    EQT_MAX,
    INSOL_MIN,
    INSOL_MAX,
    RADE_MIN,
    RADE_MAX,
    APP_HABITABLE_MARKER_SIZE,
    APP_UNINHABITABLE_MARKER_SIZE,
)

# Suppress specific RuntimeWarnings from sklearn
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.utils.extmath"
)
# Suppress specific UserWarning for KMeans memory leak on Windows with MKL (if applicable)
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


def determine_habitability(row):
    required_hab_cols = ["pl_eqt", "pl_insol", "pl_rade"]
    if any(pd.isna(row[col]) for col in required_hab_cols):
        return False
    try:
        return (
            EQT_MIN <= row["pl_eqt"] <= EQT_MAX
            and INSOL_MIN <= row["pl_insol"] <= INSOL_MAX
            and RADE_MIN <= row["pl_rade"] <= RADE_MAX
        )
    except (
        TypeError
    ):  # Handles cases where a value might not be comparable (e.g., string if data isn't clean)
        return False


def get_non_habitable_reasons_str(row):
    if row["is_habitable"]:
        return ""
    reasons = []
    # Using .get(col) for robustness in case a column is unexpectedly missing from a row
    # although apply works row-wise so columns should exist if in df.
    if pd.isna(row.get("pl_rade")):
        reasons.append("Radius: Data missing")
    elif not (
        RADE_MIN <= row.get("pl_rade", np.nan) <= RADE_MAX
    ):  # np.nan if missing to ensure comparison fails safely
        reasons.append(
            f"Radius: {row.get('pl_rade'):.2f} ER (Not in [{RADE_MIN:.1f}, {RADE_MAX:.1f}])"
        )
    if pd.isna(row.get("pl_eqt")):
        reasons.append("Eq. Temp: Data missing")
    elif not (EQT_MIN <= row.get("pl_eqt", np.nan) <= EQT_MAX):
        temp_reason = f"Eq. Temp: {row.get('pl_eqt'):.1f}K "
        if row.get("pl_eqt", np.nan) < EQT_MIN:
            temp_reason += f"(Too low, < {EQT_MIN}K)"
        else:
            temp_reason += f"(Too high, > {EQT_MAX}K)"
        reasons.append(temp_reason)
    if pd.isna(row.get("pl_insol")):
        reasons.append("Insolation: Data missing")
    elif not (INSOL_MIN <= row.get("pl_insol", np.nan) <= INSOL_MAX):
        insol_reason = f"Insolation: {row.get('pl_insol'):.2f} EF "
        if row.get("pl_insol", np.nan) < INSOL_MIN:
            insol_reason += f"(Too low, < {INSOL_MIN:.2f}x)"
        else:
            insol_reason += f"(Too high, > {INSOL_MAX:.2f}x)"
        reasons.append(insol_reason)
    if not reasons and not row["is_habitable"]:
        return "Reasons not habitable: Fails criteria (unspecified details)"
    if reasons:
        return "Reasons not habitable:\n" + "\n".join(f"- {r}" for r in reasons)
    return ""


def format_value(value, precision_type):
    if pd.isna(value):
        return "N/A"
    if precision_type == "int":
        try:
            return f"{int(value)}"
        except (ValueError, TypeError):
            return "N/A"  # If value cannot be converted
    if isinstance(precision_type, int):
        try:
            return f"{float(value):.{precision_type}f}"
        except (ValueError, TypeError):
            return "N/A"
    return str(value)


@st.cache_data
def load_and_prepare_data(file_path="data/Planetary-Systems-May-20-2025_clean.csv"):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(  # Acceptable in cached func if it's high-level data loading for the app
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

    df = df.dropna(subset=coord_cols).copy()  # Essential for coordinates

    if df.empty:
        st.warning("DataFrame is empty after dropping NaNs in coordinate columns.")
        return None

    df["sy_dist_ly"] = df["sy_dist"] * PARSEC_TO_LY

    earth_data = {
        "pl_name": "Earth (Sol System)",
        "sy_dist": 0.0,
        "sy_dist_ly": 0.0,
        "ra": 0.0,
        "dec": 0.0,
        "pl_rade": 1.0,
        "pl_eqt": 288.0,
        "pl_insol": 1.0,
        "pl_dens": 5.51,
        "glon": np.nan,
        "glat": np.nan,
    }
    if not df["pl_name"].astype(str).str.contains("Earth", case=False).any():
        earth_df_row = pd.DataFrame([earth_data])
        df = pd.concat([df, earth_df_row], ignore_index=True)
    else:  # Ensure Earth data is up-to-date if it exists
        earth_idx = df[
            df["pl_name"].astype(str).str.contains("Earth", case=False)
        ].index
        if not earth_idx.empty:
            for col, val in earth_data.items():
                df.loc[earth_idx, col] = val

    df["is_habitable"] = df.apply(determine_habitability, axis=1)
    df["non_habitable_reasons"] = df.apply(get_non_habitable_reasons_str, axis=1)
    df["non_habitable_reasons_html"] = df["non_habitable_reasons"].str.replace(
        "\n", "<br>", regex=False
    )
    df["x"], df["y"], df["z"] = spherical_to_cartesian(
        df["ra"], df["dec"], df["sy_dist_ly"]
    )

    # Use .get() in lambda for base_hover_text for robustness if a column is missing
    df["base_hover_text"] = df.apply(
        lambda row: f"""<b>{row.get('pl_name', 'N/A')}</b><br>
Distance: {format_value(row.get('sy_dist_ly'), 2)} ly<br>
Radius: {format_value(row.get('pl_rade'), 2)} ER<br>
Density: {format_value(row.get('pl_dens'), 2)} g/cm³<br>
Eq. Temp: {format_value(row.get('pl_eqt'), 1)} K<br>
Insolation: {format_value(row.get('pl_insol'), 2)} EF<br>
Habitable (Criteria): {row.get('is_habitable', 'N/A')}<br>
{row.get('non_habitable_reasons_html', '') if not row.get('is_habitable', True) and pd.notna(row.get('non_habitable_reasons_html')) else ''}
""".strip(),
        axis=1,
    )

    df["app_marker_size"] = df["is_habitable"].apply(
        lambda hab: APP_HABITABLE_MARKER_SIZE if hab else APP_UNINHABITABLE_MARKER_SIZE
    )
    # Ensure Earth gets habitable marker size and status
    earth_idx_final = df[df["pl_name"] == "Earth (Sol System)"].index
    if not earth_idx_final.empty:
        df.loc[earth_idx_final, "app_marker_size"] = APP_HABITABLE_MARKER_SIZE
        df.loc[earth_idx_final, "is_habitable"] = True

    return df


@st.cache_data
def run_clustering(_df_input, n_clusters_requested, cluster_features_list):
    df = _df_input.copy()
    # Initialize cluster column: -1 signifies "not clustered" or "unclusterable"
    df["cluster"] = -1

    if not cluster_features_list:
        # No features selected by user, all data points remain in cluster -1
        return df, 1  # Effectively 1 group of "unclustered" items (-1)

    # Ensure all selected features exist in the DataFrame columns
    missing_cols = [col for col in cluster_features_list if col not in df.columns]
    if missing_cols:
        # This indicates an issue with feature_options_map or data loading.
        # All data points will remain cluster -1.
        # Silently return; calling function can inform user if needed.
        return df, 1

    # Prepare data for clustering: select features
    features_for_clustering_df = df[cluster_features_list].copy()

    # Drop rows where *any* of the selected clustering features are NaN
    features_for_clustering_df.dropna(subset=cluster_features_list, inplace=True)

    valid_indices_for_clustering = features_for_clustering_df.index

    if features_for_clustering_df.empty:
        # No data points left after removing NaNs for selected features
        return df, 1  # All original data points remain cluster -1

    features_for_clustering_values = features_for_clustering_df.values

    if len(features_for_clustering_values) < 2:
        if len(features_for_clustering_values) == 1:  # Single valid data point
            df.loc[valid_indices_for_clustering, "cluster"] = (
                0  # Assign it to cluster 0
            )
        return df, 1  # Single cluster type (0 or -1)

    # Check if all feature values are identical (no variance)
    # Need to construct a DataFrame to use .nunique()
    if (
        pd.DataFrame(features_for_clustering_values, columns=cluster_features_list)
        .nunique()
        .max()
        == 1
    ):
        df.loc[valid_indices_for_clustering, "cluster"] = (
            0  # All same, assign to cluster 0
        )
        return df, 1

    actual_n_clusters = max(
        1, min(n_clusters_requested, len(features_for_clustering_values))
    )

    if (
        actual_n_clusters == 1
    ):  # Only one cluster requested or possible for the valid data
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features_for_clustering_values)
    except ValueError:  # Should be caught by nunique check, but as a safeguard
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    if np.any(np.isnan(scaled_features)) or np.any(np.isinf(scaled_features)):
        # This is highly unlikely if NaNs were dropped and data is finite.
        df.loc[valid_indices_for_clustering, "cluster"] = 0
        return df, 1

    try:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(scaled_features)
        df.loc[valid_indices_for_clustering, "cluster"] = cluster_labels
    except Exception:
        df.loc[valid_indices_for_clustering, "cluster"] = (
            0  # Fallback for prepared data if KMeans errors
        )
        actual_n_clusters = (
            1  # Reflects that clustering effectively resulted in one group (0s and -1s)
        )

    return df, actual_n_clusters
