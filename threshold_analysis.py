import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# 1. Load Merged Features Dataset
csv_path = Path("mandi_landslides_features_merged.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Could not find {csv_path}. Ensure feature extraction script was run successfully.")

print(f"[*] Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# 2. Setup Output Directory
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)

# 3. Define Threshold Variables
x_col = 'tp_3_day_cum'  # 3-day cumulative antecedent precipitation
y_col = 'swvl1_event'   # Top-layer volumetric soil water content

if x_col in df.columns and y_col in df.columns:
    plt.figure(figsize=(9, 6))
    
    # Scatter plot of landslide events
    sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        color='firebrick', 
        s=90, 
        edgecolor='black', 
        alpha=0.85,
        label='Mandi Landslide Events'
    )
    
    # Calculate baseline lower-bound thresholds (using lower quartile or characteristic minimum trigger levels)
    x_thresh = df[x_col].quantile(0.20)
    y_thresh = df[y_col].quantile(0.20)
    
    plt.axvline(x=x_thresh, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Min Rain Threshold (~{x_thresh:.4f} m)')
    plt.axhline(y=y_thresh, color='dodgerblue', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Min Soil Saturation (~{y_thresh:.4f})')
    
    # Annotate critical zone
    plt.title("Empirical Rainfall vs. Soil Moisture Trigger Thresholds (Mandi)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("3-Day Cumulative Antecedent Precipitation (m)", fontsize=11)
    plt.ylabel("Root-Zone Soil Moisture [swvl1] (m³/m³)", fontsize=11)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    plot_path = output_dir / "rainfall_soil_moisture_threshold.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Threshold plot successfully saved to: {plot_path}")
else:
    print(f"[!] Required columns ({x_col}, {y_col}) not found. Available columns: {df.columns.tolist()}")