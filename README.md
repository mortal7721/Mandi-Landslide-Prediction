# 🏔️ Hydro-Meteorological Landslide Susceptibility & Threshold Analysis in the Mandi Himalayas

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Reanalysis-ERA5--Land%20%7C%20NASA%20GLC-teal.svg)](https://cds.climate.copernicus.eu/)
[![Institution](https://img.shields.io/badge/IIT%20Mandi-Applied%20Hydroclimatology-darkblue.svg)](https://www.iitmandi.ac.in/)


An integrated **hydro-climatological and machine learning framework** developed to investigate antecedent rainfall and subsurface soil moisture triggers for landslides in the **Mandi–Hamirpur region of the North-Western Himalayas** ($31.2^\circ\text{N}–32.3^\circ\text{N}, 76.4^\circ\text{E}–77.5^\circ\text{E}$).

> **Academic Context:** Course Project for *Applied Hydroclimatology (CE-562)* at the School of Civil and Environmental Engineering (SCENE), **Indian Institute of Technology (IIT) Mandi**, under the guidance of **Dr. Vivek Gupta**.

---

## 📌 Table of Contents
1. [Motivation & Geotechnical Physics](#-motivation--geotechnical-physics)
2. [Data Architecture](#-data-architecture)
3. [Repository Structure](#-repository-structure)
4. [End-to-End Code Workflow & Explanation](#-end-to-end-code-workflow--explanation)
5. [Key Findings & Empirical Thresholds](#-key-findings--empirical-thresholds)
6. [Installation & Usage](#-installation--usage)
7. [References](#-references)

---

## 🔬 Motivation & Geotechnical Physics

Traditional landslide early-warning systems rely heavily on empirical rainfall Intensity–Duration ($I\text{-}D$) power-law curves ($I = \alpha \cdot D^{-\beta}$). However, in steep Himalayan catchments, slope failure is rarely caused by isolated single-day rainfall peaks alone. Instead, it follows a **two-stage unsaturated soil mechanics progression** (Fredlund & Rahardjo, 1993):

$$\tau = c' + (\sigma_n - u_a)\tan\phi' + (u_a - u_w)\tan\phi^b$$

1. **Pre-Conditioning Phase:** Prolonged multi-day antecedent rainfall (`3-day` and `7-day` cumulative windows) infiltrates the soil profile (`swvl1` through `swvl4`), progressively eliminating soil matric suction $(u_a - u_w)$ and reducing unsaturated shear strength.
2. **Triggering Phase:** Subsequent rainfall acting on an already saturated soil column ($\approx 0.40–0.50\text{ m}^3/\text{m}^3$) generates rapid positive pore-water pressure spikes along the soil–bedrock interface, initiating shallow planar slides and debris flows.

---

## 🛰️ Data Architecture

This repository synthesizes three authoritative geospatial and hydrometeorological datasets:

| Dataset | Source | Spatial / Temporal Resolution | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **Landslide Inventory** | NASA Global Landslide Catalog (GLC) | Point coordinates & timestamps | 46 verified historical landslide events ($\text{Class } 1$) |
| **Reanalysis Forcing** | ECMWF ERA5-Land | $\sim 9\text{ km}$ grid / Hourly | Precipitation (`tp`), Temperature (`t2m`), Solar Radiation (`ssrd`), and 4-layer Volumetric Soil Water (`swvl1`–`swvl4`) |
| **Topography** | SRTM Digital Elevation Model | $30\text{ m}$ raster | Elevation ($Z$), slope gradient proxies, and terrain relief mapping |

---

## 📂 Repository Structure

```text
mandi-landslide-hydroclimatology/
│
├── data/
│   └── raw/                                  # Raw NASA GLC CSV, ERA5-Land .nc files, SRTM .tar.gz
│
├── eda_plots/                                # Generated high-resolution figures (300 DPI)
│   ├── study_area_dem_map.png
│   ├── mandi_landslide_spatial_map.png
│   ├── antecedent_precipitation_distribution.png
│   ├── soil_moisture_profile_boxplot.png
│   ├── correlation_heatmap.png
│   ├── antecedent_moisture_precipitation.png
│   ├── feature_importance_rf.png
│   └── model_validation_curves_simple.png
│
├── 1_process_raw_data.py                     # Filters NASA GLC for Mandi & inspects ERA5 NetCDF files
├── 2_extract_features.py                     # Spatio-temporal matching & cumulative rainfall engineering
├── 3_plot_dem_map.py                         # Generates SRTM 30m elevation map of Mandi-Hamirpur
├── 4_spatial_mapping.py                      # Maps 46 historical landslides graded by 3-day rainfall
├── 5_eda_analysis.py                         # Statistical summaries, KDE distributions, boxplots & heatmaps
├── 6_threshold_analysis.py                   # Derives lower-bound soil saturation & rainfall thresholds
├── 7_simple_id_threshold.py                  # Evaluates classical log-log I-D power-law baseline
├── 8_model_feature_importance.py             # Trains Random Forest classifier & ranks feature importance
├── 9_simple_model_validation.py              # Executes 5-Fold Stratified Cross-Validation (ROC-AUC & PR)
└── README.md                                 # Complete project documentation
```

---

## ⚙️ End-to-End Code Workflow & Explanation

### **Phase 1: Data Filtering & Spatio-Temporal Feature Engineering**

#### 1. `process_raw_data.py`
* **Purpose:** Ingests the global NASA landslide database (`Global_Landslide_Catalog_Export_rows.csv`) and filters records spatially for the Mandi–Hamirpur bounding box ($31.2^\circ\text{N}–32.3^\circ\text{N}, 76.4^\circ\text{E}–77.5^\circ\text{E}$).
* **Output:** Generates `mandi_landslides_processed.csv` (46 verified events) and verifies variable dimensions across raw ERA5-Land NetCDF (`.nc`) archives using `h5py`.

#### 2. `extract_features.py`
* **Purpose:** Bridges discrete landslide timestamps/coordinates with continuous gridded ERA5-Land reanalysis fields using `xarray`.
* **Key Operations:**
  * Matches each landslide coordinate to the nearest ERA5-Land grid centroid.
  * Extracts instantaneous event states (`tp_event`, `t2m_event`, `d2m_event`, `ssrd_event`) and four-layer soil moisture (`swvl1_event` to `swvl4_event` covering $0–289\text{ cm}$ depth).
  * Integrates **3-day (`tp_3_day_cum`)** and **7-day (`tp_7_day_cum`)** cumulative antecedent precipitation windows prior to failure.
* **Output:** Saves the master feature matrix `mandi_landslides_features_merged.csv`.

---

### **Phase 2: Topographic & Exploratory Data Analysis (EDA)**

#### 3. `plot_dem_map.py` & `spatial_mapping.py`
* **Purpose:** Visualizes the physical terrain and hazard distribution. `plot_dem_map.py` reads the SRTM 30m GeoTIFF (`output_SRTMGL1.tif`) to map elevation gradients ($500\text{ m}$ Beas valleys to $>3,000\text{ m}$ ridges), while `spatial_mapping.py` plots the 46 landslide locations color-coded by 3-day antecedent rainfall intensity.

#### 4. `eda_analysis.py`
* **Purpose:** Computes summary statistics (mean, median, IQR) and generates core hydro-meteorological diagnostics:
  * **Rainfall Distributions:** Histograms + KDE curves comparing 3-day (mean $53.7\text{ mm}$) vs. 7-day (mean $98.7\text{ mm}$) rainfall accumulation.
  * **Subsurface Saturation Profile:** Boxplots across `swvl1` ($0–7\text{ cm}$), `swvl2` ($7–28\text{ cm}$), `swvl3` ($28–100\text{ cm}$), and `swvl4` ($100–289\text{ cm}$), demonstrating full-column saturation (medians $> 0.42\text{ m}^3/\text{m}^3$).
  * **Correlation Matrix:** Pearson correlation heatmap quantifying precursor collinearity (e.g., $r = 0.67$ between 3-day and 7-day rainfall).

---

### **Phase 3: Empirical Trigger Thresholds**

#### 5. `threshold_analysis.py` & `simple_id_threshold.py`
* **Purpose:** Establishes physical lower-bound trigger envelopes by plotting `tp_3_day_cum` against surface volumetric soil moisture (`swvl1_event`), contrasting modern soil-wetness coupling against classical log-log Intensity–Duration ($I\text{-}D$) regression (`simple_id_threshold.py`).

---

### **Phase 4: Machine Learning Susceptibility Modeling**

#### 6. `model_feature_importance.py`
* **Purpose:** Integrates SRTM DEM elevation/slope proxies with ERA5-Land hydro-meteorological precursors. Pairs historical landslide events ($\text{Class } 1$) with baseline non-failure conditions ($\text{Class } 0$) to train a **100-tree Random Forest Classifier** and extract Gini feature importance rankings.

#### 7. `simple_model_validation.py`
* **Purpose:** Rigorously evaluates model generalization using **5-Fold Stratified Cross-Validation** (`StratifiedKFold`), generating multi-fold **Receiver Operating Characteristic (ROC-AUC)** and **Precision–Recall (PR)** curves.

---

## 📊 Key Findings & Empirical Thresholds

* **Critical Saturation Threshold:** Slope failures predominantly cluster once surface volumetric soil moisture (`swvl1`) exceeds **$\approx 0.3959\text{ m}^3/\text{m}^3$** (concentrated between $0.40$ and $0.50\text{ m}^3/\text{m}^3$).
* **Minimum Antecedent Rainfall:** A lower-bound 3-day cumulative rainfall threshold of **$\approx 0.0161\text{ m}$ ($16.1\text{ mm}$)** is required to transition saturated Himalayan slopes into failure.
* **Dominant Precursors:** Random Forest Gini rankings prove that **7-day (`tp_7_day_cum`) and 3-day (`tp_3_day_cum`) cumulative rainfall** account for **$>60\%$** of predictive importance, followed by downward solar radiation (`ssrd_event`, governing evaporative drying) and root-zone soil moisture (`swvl1`–`swvl2`).
* **Model Discrimination:** Across 5-Fold Stratified Cross-Validation, the classifier achieved **ROC-AUC scores of $0.93–1.00$** and **Average Precision (AP) scores of $0.94–1.00$**, confirming minimal false-alarm rates.

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mandi-landslide-hydroclimatology.git
   cd mandi-landslide-hydroclimatology
   ```

2. **Install required dependencies:**
   ```bash
   pip install numpy pandas xarray netCDF4 h5py scikit-learn matplotlib seaborn rasterio tifffile
   ```

3. **Run the pipeline sequentially:**
   ```bash
   python process_raw_data.py
   python extract_features.py
   python plot_dem_map.py
   python spatial_mapping.py
   python eda_analysis.py
   python threshold_analysis.py
   python model_feature_importance.py
   python simple_model_validation.py
   ```

---

## 📚 References

* **Bogaard, T. A., & Greco, R. (2018).** Invitation for discussion: hydrological perspectives on precipitation-triggered landslides. *Hydrology and Earth System Sciences*, 22(2), 1423–1439.
* **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32.
* **Caine, N. (1980).** The rainfall intensity-duration control of shallow landslides and debris flows. *Geografiska Annaler: Series A, Physical Geography*, 62(1-2), 23–27.
* **Fredlund, D. G., & Rahardjo, H. (1993).** *Soil Mechanics for Unsaturated Soils*. John Wiley & Sons.
* **Kirschbaum, D. B., et al. (2010).** A global landslide catalog for hazardous event mapping: status, preliminary results, and future directions. *Natural Hazards*, 52(2), 561–581.
* **Muñoz-Sabater, J., et al. (2021).** ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13(9), 4349–4383.
