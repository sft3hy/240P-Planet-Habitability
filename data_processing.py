# data_processing.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st  # For st.warning/st.error in clustering if needed
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

# Suppress specific RuntimeWarnings from sklearn.utils.extmath if they occur during clustering
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.utils.extmath"
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
    except TypeError:
        return False


def get_non_habitable_reasons_str(row):
    if row["is_habitable"]:
        return ""
    reasons = []
    if pd.isna(row["pl_rade"]):
        reasons.append("Radius: Data missing")
    elif not (RADE_MIN <= row["pl_rade"] <= RADE_MAX):
        reasons.append(
            f"Radius: {row['pl_rade']:.2f} ER (Not in [{RADE_MIN:.1f}, {RADE_MAX:.1f}])"
        )
    if pd.isna(row["pl_eqt"]):
        reasons.append("Eq. Temp: Data missing")
    elif not (EQT_MIN <= row["pl_eqt"] <= EQT_MAX):
        temp_reason = f"Eq. Temp: {row['pl_eqt']:.1f}K "
        if row["pl_eqt"] < EQT_MIN:
            temp_reason += f"(Too low, < {EQT_MIN}K)"
        else:
            temp_reason += f"(Too high, > {EQT_MAX}K)"
        reasons.append(temp_reason)
    if pd.isna(row["pl_insol"]):
        reasons.append("Insolation: Data missing")
    elif not (INSOL_MIN <= row["pl_insol"] <= INSOL_MAX):
        insol_reason = f"Insolation: {row['pl_insol']:.2f} EF "
        if row["pl_insol"] < INSOL_MIN:
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
        return f"{int(value)}"
    if isinstance(precision_type, int):
        return f"{float(value):.{precision_type}f}"
    return str(value)


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

    df["is_habitable"] = df.apply(determine_habitability, axis=1)
    df["non_habitable_reasons"] = df.apply(get_non_habitable_reasons_str, axis=1)
    df["non_habitable_reasons_html"] = df["non_habitable_reasons"].str.replace(
        "\n", "<br>", regex=False
    )
    df["x"], df["y"], df["z"] = spherical_to_cartesian(
        df["ra"], df["dec"], df["sy_dist_ly"]
    )

    df["base_hover_text"] = df.apply(
        lambda row: f"""<b>{row['pl_name']}</b><br>
Distance: {format_value(row['sy_dist_ly'], 2)} ly<br>
Radius: {format_value(row['pl_rade'], 2)} ER<br>
Density: {format_value(row['pl_dens'], 2)} g/cm³<br>
Eq. Temp: {format_value(row['pl_eqt'], 1)} K<br>
Insolation: {format_value(row['pl_insol'], 2)} EF<br>
Habitable (Criteria): {row['is_habitable']}<br>
{row['non_habitable_reasons_html'] if not row['is_habitable'] and pd.notna(row['non_habitable_reasons_html']) and row['non_habitable_reasons_html'] else ''}
""".strip(),
        axis=1,
    )

    df["app_marker_size"] = df["is_habitable"].apply(
        lambda hab: APP_HABITABLE_MARKER_SIZE if hab else APP_UNINHABITABLE_MARKER_SIZE
    )
    earth_idx = df[df["pl_name"] == "Earth (Sol System)"].index
    if not earth_idx.empty:
        df.loc[earth_idx, "app_marker_size"] = APP_HABITABLE_MARKER_SIZE
        df.loc[earth_idx, "is_habitable"] = True

    return df


@st.cache_data
def run_clustering(_df_input, n_clusters_requested):
    df = _df_input.copy()
    features_for_clustering = df[["x", "y", "z"]].copy()

    if features_for_clustering.empty or len(features_for_clustering) < 2:
        df["cluster"] = 0
        return df, 1

    # Check if all feature values are identical (no variance)
    if features_for_clustering.nunique().max() == 1:
        df["cluster"] = 0
        return df, 1

    actual_n_clusters = max(1, min(n_clusters_requested, len(features_for_clustering)))
    if actual_n_clusters == 1:  # KMeans requires n_clusters >= 2 if data points > 1
        df["cluster"] = 0
        return df, 1

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)

    # Check for NaN/Inf values after scaling, which can cause KMeans to fail
    if np.any(np.isnan(scaled_features)) or np.any(np.isinf(scaled_features)):
        st.warning(
            "Warning: Encountered NaN/Inf values in features after scaling. Assigning default cluster."
        )
        df["cluster"] = 0
        return df, 1

    try:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init="auto")
        df["cluster"] = kmeans.fit_predict(scaled_features)
    except Exception as e:
        st.error(f"Clustering error: {e}. Assigning default cluster.")
        df["cluster"] = 0
        return df, 1  # Return 1 as actual_n_clusters if clustering fails

    for index, row in df.iterrows():
        if row["is_habitable"]:
            print(row["pl_name"])
    return df, actual_n_clusters
