import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile

# Ensure output directory exists
os.makedirs("eda_plots", exist_ok=True)

# Read raw elevation array from GeoTIFF
dem = tifffile.imread("output_SRTMGL1.tif").astype(float)

# Mask invalid or nodata elevation values (e.g., < 0 or > 9000m)
dem[(dem <= 0) | (dem > 9000)] = np.nan

# Study area bounding box: [lon_min, lon_max, lat_min, lat_max]
extent = [76.4, 77.5, 31.2, 32.3]

plt.figure(figsize=(8, 6), dpi=300)
img = plt.imshow(dem, cmap="terrain", extent=extent, aspect="auto")
plt.colorbar(img, label="Elevation (m above sea level)")
plt.title(
    "SRTM 30m Digital Elevation Model (Mandi-Hamirpur Study Area)",
    fontsize=11,
    fontweight="bold",
)
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("eda_plots/study_area_dem_map.png", dpi=300)
plt.show()