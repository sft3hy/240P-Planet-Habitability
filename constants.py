# constants.py

# --- Physical Constants ---
PARSEC_TO_LY = 3.26156

# --- Habitability Constraints (used globally) ---
EQT_MIN, EQT_MAX = 180, 310  # Kelvin
INSOL_MIN, INSOL_MAX = 0.25, 3.0  # Earth flux
RADE_MIN, RADE_MAX = 0.5, 3.0  # Earth radii

# --- Plotly Visual Parameters for the App ---
APP_HABITABLE_MARKER_SIZE = 8
APP_UNINHABITABLE_MARKER_SIZE = 4
APP_SUN_MARKER_SIZE = 15
HABITABLE_COLOR_STR = "rgb(0, 191, 255)"  # Deep Sky Blue
NON_HABITABLE_COLOR_STR = "rgb(200,50,50)"  # Red
LOCAL_NEIGHBORHOOD_OPACITY = 0.1
SOLAR_SYSTEM_OPACITY = 0.5

# --- Camera Control Parameters ---
CAMERA_DEFAULT_EYE_X = 400
CAMERA_DEFAULT_EYE_Y = 400
CAMERA_DEFAULT_EYE_Z = 250
CAMERA_EYE_FACTOR = 1
CAMERA_Z_FACTOR = 0.6
CAMERA_EFFECTIVE_DISTANCE_MIN = 200
CAMERA_EFFECTIVE_DISTANCE_MAX = 3500

# --- Legend Position ---
# Places the legend inside the plot area, at the top-right.
# yanchor='top': The 'y' coordinate refers to the top of the legend.
# y=0.99: Positions the top of the legend at 99% of the plot height (from the bottom).
# xanchor='right': The 'x' coordinate refers to the right of the legend.
# x=0.99: Positions the right of the legend at 99% of the plot width (from the left).
LEGEND_STYLE = dict(
    orientation="v",  # Vertical legend for better fit in corner
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,
    bgcolor="rgba(0,0,0,0.6)",  # Semi-transparent black background for legend
    bordercolor="rgba(255,255,255,0.5)",  # Light border for legend
    borderwidth=1,
)
