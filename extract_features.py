import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

# Define paths
data_dir = Path("data/raw")
csv_path = Path("mandi_landslides_processed.csv")

# 1. Load Mandi Landslide Data
if not csv_path.exists():
    # Check if it was saved inside data/raw/
    csv_path = data_dir / "mandi_landslides_processed.csv"

print(f"[*] Loading landslide records from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"[-] Loaded {len(df)} records. Columns found: {list(df.columns)}")

# Ensure datetime column exists (adjust 'date' or 'timestamp' based on your CSV structure)
# Looking for common date column names
date_col = next((col for col in ['date', 'datetime', 'time', 'event_date'] if col in df.columns), None)
lat_col = next((col for col in ['latitude', 'lat'] if col in df.columns), None)
lon_col = next((col for col in ['longitude', 'lon', 'long'] if col in df.columns), None)

if not all([date_col, lat_col, lon_col]):
    raise ValueError(f"Could not automatically detect date, latitude, or longitude columns. Found columns: {df.columns.tolist()}")

df[date_col] = pd.to_datetime(df[date_col])

# 2. Open ERA5 NetCDF Datasets
print("[*] Opening ERA5 NetCDF files...")
nc_files = list(data_dir.glob("*.nc"))

datasets = {}
for file in nc_files:
    ds = xr.open_dataset(file)
    # Identify variables inside the dataset
    for var in ds.data_vars:
        datasets[var] = ds
        print(f"    Loaded variable '{var}' from {file.name}")

# 3. Feature Extraction Loop
print("[*] Extracting meteorological and hydrological features for each landslide event...")

extracted_features = []

for idx, row in df.iterrows():
    event_time = row[date_col]
    lat = row[lat_col]
    lon = row[lon_col]
    
    feature_row = {
        'index': idx,
        'date': event_time,
        'latitude': lat,
        'longitude': lon
    }
    
    # Extract point data or nearest grid cell for each variable
    for var, ds in datasets.items():
        try:
            # Select nearest lat/lon and time slice up to the event date
            # Note: ERA5 coordinates are usually 'latitude', 'longitude', 'valid_time' or 'time'
            time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
            
            # Extract value at exact event time or latest available prior
            val_ds = ds[var].sel(latitude=lat, longitude=lon, method="nearest")
            
            # Get value at event time
            val_at_event = val_ds.sel({time_dim: event_time}, method="nearest").values.item()
            feature_row[f"{var}_event"] = val_at_event
            
            # If variable is total precipitation (tp), compute antecedent sums (e.g., 3-day, 7-day cumulative)
            if var == 'tp':
                # Past 3 days cumulative
                t_start_3d = event_time - pd.Timedelta(days=3)
                tp_3d = val_ds.sel({time_dim: slice(t_start_3d, event_time)}).sum().values.item()
                feature_row['tp_3_day_cum'] = tp_3d
                
                # Past 7 days cumulative
                t_start_7d = event_time - pd.Timedelta(days=7)
                tp_7d = val_ds.sel({time_dim: slice(t_start_7d, event_time)}).sum().values.item()
                feature_row['tp_7_day_cum'] = tp_7d

        except Exception as e:
            # Handle out-of-bounds or coordinate mismatch gracefully
            feature_row[f"{var}_event"] = np.nan

    extracted_features.append(feature_row)

# 4. Create Feature DataFrame & Save
features_df = pd.DataFrame(extracted_features)
output_file = "mandi_landslides_features_merged.csv"
features_df.to_csv(output_file, index=False)

print(f"\n[+] Feature extraction complete! Saved merged dataset to: {output_file}")
print(features_df.head())