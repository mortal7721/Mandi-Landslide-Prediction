import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# 1. Load Merged Features Dataset
csv_path = Path("mandi_landslides_features_merged.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Could not find {csv_path}. Ensure feature extraction was run successfully.")

print(f"[*] Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# 2. Check for required spatial columns
lat_col = next((col for col in ['latitude', 'lat'] if col in df.columns), None)
lon_col = next((col for col in ['longitude', 'lon', 'long'] if col in df.columns), None)

if not lat_col or not lon_col:
    raise ValueError("Latitude and Longitude columns not found in dataset.")

# 3. Create Spatial Distribution Plot
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10, 8))

# Scatter plot of Mandi landslides colored by 3-day antecedent precipitation
scatter = plt.scatter(
    df[lon_col], 
    df[lat_col], 
    c=df.get('tp_3_day_cum', df['tp_event']), 
    cmap='YlOrRd', 
    s=100, 
    edgecolor='black', 
    linewidth=0.8,
    alpha=0.9
)

cbar = plt.colorbar(scatter)
cbar.set_label('3-Day Cumulative Antecedent Rainfall (m)', fontsize=11, fontweight='bold')

# Map annotations & formatting
plt.title("Spatial Distribution of Mandi Landslide Events\n(North-Western Himalayas)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Longitude (°E)", fontsize=11)
plt.ylabel("Latitude (°N)", fontsize=11)

# Add gridlines for geographic context
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save plot
plot_path = output_dir / "mandi_landslide_spatial_map.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"[+] Spatial GIS map successfully saved to: {plot_path}")