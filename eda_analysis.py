import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# 1. Load Merged Features Dataset
csv_path = Path("mandi_landslides_features_merged.csv")
if not csv_path.exists():
  raise FileNotFoundError(
      f"Could not find {csv_path}. Make sure feature extraction ran"
      " successfully."
  )

print(f"[*] Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# Ensure datetime conversion
if "date" in df.columns:
  df["date"] = pd.to_datetime(df["date"])

print(f"[-] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

# 2. Summary Statistics
print("=" * 50)
print(" KEY SUMMARY STATISTICS ")
print("=" * 50)
key_cols = [
    c
    for c in [
        "tp_event",
        "tp_3_day_cum",
        "tp_7_day_cum",
        "swvl1_event",
        "swvl4_event",
        "t2m_event",
        "ssrd_event",
    ]
    if c in df.columns
]
summary_stats = df[key_cols].describe().T[
    ["count", "mean", "std", "min", "50%", "max"]
]
summary_stats.rename(columns={"50%": "median"}, inplace=True)
print(summary_stats.to_string())
print("=" * 50 + "\n")

# 3. Create Visualizations Directory
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)

# Plot 1: Cumulative Antecedent Precipitation Distribution
if "tp_3_day_cum" in df.columns and "tp_7_day_cum" in df.columns:
  plt.figure(figsize=(10, 5))
  sns.histplot(
      df["tp_3_day_cum"],
      color="blue",
      label="3-Day Cumulative Precipitation",
      kde=True,
      alpha=0.5,
  )
  sns.histplot(
      df["tp_7_day_cum"],
      color="orange",
      label="7-Day Cumulative Precipitation",
      kde=True,
      alpha=0.3,
  )
  plt.title(
      "Distribution of Antecedent Precipitation Preceding Mandi Landslides",
      fontsize=11,
      fontweight="bold",
  )
  plt.xlabel("Precipitation (m)")
  plt.ylabel("Frequency / Count")
  plt.legend()
  plt.tight_layout()
  plt.savefig(
      output_dir / "antecedent_precipitation_distribution.png", dpi=300
  )
  plt.close()
  print("[+] Saved: eda_plots/antecedent_precipitation_distribution.png")

# Plot 2: Soil Moisture Profile across Layers (swvl1 to swvl4)
soil_cols = [
    c
    for c in [
        "swvl1_event",
        "swvl2_event",
        "swvl3_event",
        "swvl4_event",
    ]
    if c in df.columns
]
if soil_cols:
  plt.figure(figsize=(10, 6))
  soil_melted = df[soil_cols].melt(
      var_name="Soil Layer", value_name="Soil Water Content"
  )
  soil_melted["Soil Layer"] = (
      soil_melted["Soil Layer"].str.replace("_event", "").str.upper()
  )
  sns.boxplot(
      x="Soil Layer",
      y="Soil Water Content",
      data=soil_melted,
      palette="Blues",
  )
  plt.title(
      "Soil Moisture Profile Across Depth Layers During Landslide Events",
      fontsize=11,
      fontweight="bold",
  )
  plt.xlabel("Soil Layer Depth")
  plt.ylabel("Volumetric Soil Water Content (m³/m³)")
  plt.tight_layout()
  plt.savefig(output_dir / "soil_moisture_profile_boxplot.png", dpi=300)
  plt.close()
  print("[+] Saved: eda_plots/soil_moisture_profile_boxplot.png")

# Plot 3: Antecedent Precipitation vs Surface Soil Moisture (New hydrology integration)
if (
    "tp_3_day_cum" in df.columns
    and "swvl1_event" in df.columns
    and "tp_7_day_cum" in df.columns
):
  plt.figure(figsize=(8, 6), dpi=300)
  scatter = plt.scatter(
      df["tp_3_day_cum"],
      df["swvl1_event"],
      c=df["tp_7_day_cum"],
      cmap="viridis",
      s=70,
      edgecolors="black",
      alpha=0.85,
  )
  plt.colorbar(scatter, label="7-Day Cumulative Precipitation (m)")
  plt.title(
      "Antecedent Precipitation & Surface Soil Moisture (Mandi Landslides)",
      fontsize=11,
      fontweight="bold",
  )
  plt.xlabel(
      "3-Day Cumulative Precipitation (`tp_3_day_cum` - meters)", fontsize=10
  )
  plt.ylabel(
      "Surface Soil Moisture (`swvl1_event` volumetric water content)",
      fontsize=10,
  )
  plt.grid(True, linestyle="--", alpha=0.4)
  plt.tight_layout()
  plt.savefig(output_dir / "antecedent_moisture_precipitation.png", dpi=300)
  plt.close()
  print("[+] Saved: eda_plots/antecedent_moisture_precipitation.png")

# Plot 4: Correlation Heatmap
if len(key_cols) > 1:
  plt.figure(figsize=(10, 8))
  corr = df[key_cols].corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  sns.heatmap(
      corr,
      annot=True,
      fmt=".2f",
      cmap="coolwarm",
      mask=mask,
      linewidths=0.5,
      cbar_kws={"label": "Pearson Correlation"},
  )
  plt.title(
      "Correlation Matrix of Hydro-Meteorological Precursors",
      fontsize=11,
      fontweight="bold",
  )
  plt.tight_layout()
  plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
  plt.close()
  print("[+] Saved: eda_plots/correlation_heatmap.png")

print(
    f"\n[+] Updated EDA complete! All plots saved in the '{output_dir}' folder."
)