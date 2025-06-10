# Exoplanet Habitability Explorer 🌌

An interactive Streamlit web application for visualizing and analyzing exoplanet data from the NASA Exoplanet Archive. This tool allows users to explore a 3D map of known exoplanets, analyze their habitability potential based on key stellar and planetary parameters, and apply machine learning techniques like K-Means, LDA, and PCA for deeper insights.


*(Above: A demonstration of the interactive 3D classification map and the PCA analysis page.)*

---

## 🚀 Features

*   **Exoplanet Classification Map**: An interactive 3D visualization of exoplanetary systems using Plotly. Planets are colored by a custom habitability classification, and hovering over a point reveals detailed data.
*   **Habitability Analysis & Insights**: A dashboard view summarizing the classification counts, highlighting key findings, and displaying pre-generated distribution plots for the 8 key habitability parameters.
*   **K-Means Cluster Analysis**: An unsupervised learning tool to group exoplanets into clusters based on user-selected physical parameters. The number of clusters (K) is interactive.
*   **Linear Discriminant Analysis (LDA)**: A supervised dimensionality reduction technique that visualizes the separability of the predefined habitability classes.
*   **Principal Component Analysis (PCA)**: An unsupervised dimensionality reduction tool to explore the primary sources of variance in the exoplanet data. Includes multiple interactive and static plot options (2D, 3D, and projections).
*   **Dynamic UI**: All controls are neatly organized in a sidebar, allowing users to switch between analysis pages and configure parameters without cluttering the main view.

---

## 📊 Data Source

The primary dataset used in this application is a cleaned CSV file derived from the **NASA Exoplanet Archive**.

-   **File**: `data/Planetary-Systems-May-20-2025_clean.csv`
-   **Access Date**: May 20, 2025
-   **Static Images**: The "Habitability Analysis" page uses pre-generated plots stored in the `Images/` directory to display parameter distributions.

---

## 🛠️ Technical Stack

*   **Language**: Python
*   **Web Framework**: Streamlit
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: Scikit-learn
*   **Data Visualization**: Plotly, Matplotlib

---

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/exoplanet-explorer.git
    cd exoplanet-explorer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A `requirements.txt` file should be created with the following content:*
    ```txt
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    matplotlib
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application should now be open and running in your web browser.

---

## 🔬 How It Works

The application's logic is divided into three main components: data processing, visualization, and the user interface.

### 1. Data Processing (`data_processing.py`)
-   **Loading**: `load_and_prepare_data` reads the CSV, converts celestial coordinates to a 3D Cartesian system, and injects Earth's data for reference.
-   **Classification**: `classify_exoplanet` assigns each planet to one of five categories ("Excellent Candidate", "Good Planet, Poor Star", etc.) based on whether its key planetary and stellar parameters fall within predefined "habitable" ranges stored in `constants.py`.
-   **Machine Learning**:
    -   `run_clustering`: Applies K-Means clustering after scaling user-selected features.
    -   `run_lda_analysis`: Performs LDA using the habitability classification as the target variable. It handles data scaling and calculates LDA components and loadings.
    -   `perform_pca_analysis`: Imputes missing values, scales the 8 key features, and runs PCA to generate the principal components.

### 2. Visualization (`plotting.py`)
-   This module contains all functions responsible for creating plots.
-   `create_base_figure` generates the foundational 3D scene with the Sun and a reference sphere for the local neighborhood.
-   Each analysis page in `app.py` calls a corresponding `display_*` function from `plotting.py`.
-   Plotly is used for all interactive charts, enabling features like zoom, pan, and detailed hover-over text.
-   Matplotlib is used to generate the static 2D PCA plot, demonstrating an alternative plotting backend.

### 3. Application Flow (`app.py`)
-   The `main()` function serves as the entry point.
-   It initializes the page layout and sidebar.
-   The main dataset is loaded once using `st.cache_data` for efficiency.
-   A radio button in the sidebar acts as a router, conditionally calling the appropriate `display_*` function from `plotting.py` based on the user's selection.
-   Error handling is in place to manage cases where data loading fails or specific analyses cannot be run.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
