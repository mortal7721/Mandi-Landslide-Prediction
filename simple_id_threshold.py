import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set clean plotting style
sns.set_theme(style="whitegrid")

# 1. Load your merged landslide dataset
csv_path = Path("mandi_landslides_features_merged.csv")
print(f"[*] Loading data from {csv_path}")
df = pd.read_csv(csv_path)

# 2. Estimate Intensity (I) and Duration (D) for each landslide event
# Intensity = Total precipitation divided by assumed storm duration (e.g., 24 hours)
# Duration = Time window (we use 3-day cumulative window = 72 hours as an example)
df['duration_hours'] = 72.0  
df['intensity_mm_hr'] = (df['tp_3_day_cum'] * 1000) / df['duration_hours'] # convert meters to mm

# Drop missing values to keep the math clean
df_clean = df.dropna(subset=['intensity_mm_hr', 'duration_hours'])

# 3. Fit a Power-Law Curve: Intensity = alpha * Duration^(-beta)
# In log-log space, this becomes a straight line: log(I) = log(alpha) - beta * log(D)
log_D = np.log(df_clean['duration_hours'])
log_I = np.log(df_clean['intensity_mm_hr'])

# Linear fit (slope and intercept)
slope, intercept = np.polyfit(log_D, log_I, 1)
alpha = np.exp(intercept)
beta = -slope

print(f"[+] Fitted Threshold Equation: I = {alpha:.3f} * D^(-{beta:.3f})")

# 4. Plot the Intensity-Duration Threshold Curve
plt.figure(figsize=(9, 6))

# Plot actual landslide events
plt.scatter(df_clean['duration_hours'], df_clean['intensity_mm_hr'], 
            color='firebrick', s=80, edgecolor='black', label='Mandi Landslides')

# Generate curve points for the threshold line
D_line = np.linspace(df_clean['duration_hours'].min(), df_clean['duration_hours'].max(), 100)
I_line = alpha * (D_line ** (-beta))

# Draw the threshold line
plt.plot(D_line, I_line, color='blue', linestyle='--', linewidth=2, 
         label=f'Threshold: I = {alpha:.2f} D^(-{beta:.2f})')

plt.title("Rainfall Intensity-Duration (I-D) Threshold for Mandi Landslides", fontsize=12, fontweight='bold')
plt.xlabel("Effective Rainfall Duration (Hours)", fontsize=11)
plt.ylabel("Mean Rainfall Intensity (mm/h)", fontsize=11)
plt.legend()
plt.tight_layout()

# Save the plot
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "intensity_duration_threshold_simple.png", dpi=300)
plt.close()
print("[+] Saved clean I-D threshold plot to eda_plots/intensity_duration_threshold_simple.png")