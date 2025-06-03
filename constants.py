# constants.py

# --- Physical Constants ---
PARSEC_TO_LY = 3.26156

# --- Habitability Thresholds (from the paper's "very relaxed" criteria) ---
HABITABILITY_THRESHOLDS = {
    # Planetary parameters
    "pl_rade": {"min": 0.4, "max": 3.0, "name": "Planet Radius", "unit": "Earth radii"},
    "pl_eqt": {"min": 130, "max": 400, "name": "Equilibrium Temperature", "unit": "K"},
    "pl_insol": {
        "min": 0.1,
        "max": 3.0,
        "name": "Insolation Flux",
        "unit": "Earth flux",
    },
    "pl_dens": {"min": 2.5, "max": 10.0, "name": "Planet Density", "unit": "g/cm³"},
    # Stellar parameters
    "st_teff": {
        "min": 3800,
        "max": 7200,
        "name": "Stellar Eff. Temperature",
        "unit": "K",
    },
    "st_rad": {"min": 0.4, "max": 1.8, "name": "Stellar Radius", "unit": "Solar radii"},
    "st_mass": {"min": 0.3, "max": 1.8, "name": "Stellar Mass", "unit": "Solar masses"},
    "st_met": {
        "min": -0.6,
        "max": 0.6,
        "name": "Stellar Metallicity",
        "unit": "[Fe/H] dex",
    },
}

# Mapping CSV columns to key parameters and their properties for plotting
KEY_PARAMETERS_MAP = {
    "pl_rade": {"name": "Planet Radius", "unit": "Earth Radii", "type": "planetary"},
    "pl_eqt": {"name": "Equilibrium Temperature", "unit": "K", "type": "planetary"},
    "pl_insol": {"name": "Insolation Flux", "unit": "Earth Flux", "type": "planetary"},
    "pl_dens": {"name": "Planet Density", "unit": "g/cm³", "type": "planetary"},
    "st_teff": {"name": "Stellar Eff. Temp.", "unit": "K", "type": "stellar"},
    "st_rad": {"name": "Stellar Radius", "unit": "Solar Radii", "type": "stellar"},
    "st_mass": {"name": "Stellar Mass", "unit": "Solar Masses", "type": "stellar"},
    "st_met": {"name": "Stellar Metallicity", "unit": "[Fe/H] dex", "type": "stellar"},
}

# --- Plotly Visual Parameters ---
# Marker Sizes
DEFAULT_PLANET_MARKER_SIZE = 6
EXCELLENT_CANDIDATE_MARKER_SIZE = 9
SUN_MARKER_SIZE = 12
EARTH_MARKER_SIZE = 9  # Same as excellent candidate for emphasis

# Colors for Classification Categories
CLASSIFICATION_COLORS = {
    "Excellent Candidate": "gold",
    "Good Planet, Poor Star": "lightgreen",
    "Good Star, Poor Planet": "lightblue",
    "Poor Candidate": "grey",
    "Unclassified (Missing Data)": "darkslategrey",  # Changed for better visibility
}
# Hover styling
HOVER_FONT_COLOR = "white"
HOVER_BG_COLORS = {  # Background color for hover based on category
    "Excellent Candidate": "darkgoldenrod",
    "Good Planet, Poor Star": "darkgreen",
    "Good Star, Poor Planet": "darkblue",
    "Poor Candidate": "dimgray",
    "Unclassified (Missing Data)": "black",
}


LOCAL_NEIGHBORHOOD_OPACITY = 0.1
SOLAR_SYSTEM_OPACITY = 0.5  # Not currently used, but kept for potential future use

# --- Camera Control Parameters ---
CAMERA_DEFAULT_EYE_X = 400
CAMERA_DEFAULT_EYE_Y = 400
CAMERA_DEFAULT_EYE_Z = 250
CAMERA_EYE_FACTOR = 1
CAMERA_Z_FACTOR = 0.6
CAMERA_EFFECTIVE_DISTANCE_MIN = 200
CAMERA_EFFECTIVE_DISTANCE_MAX = 3500

# --- Legend Position ---
LEGEND_STYLE = dict(
    orientation="v",
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,
    bgcolor="rgba(0,0,0,0.6)",
    bordercolor="rgba(255,255,255,0.5)",
    borderwidth=1,
)
