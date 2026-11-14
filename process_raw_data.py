import os
import glob
import h5py
import pandas as pd
import numpy as np

# 1. Directory path configuration
RAW_DIR = "data/raw"

print(f"[*] Accessing raw files inside: {RAW_DIR}/")

# 2. Load and Filter Global Landslide Catalog for Himachal Pradesh / Mandi
landslide_csv_path = os.path.join(RAW_DIR, "Global_Landslide_Catalog_Export_rows.csv")

if os.path.exists(landslide_csv_path):
    df_landslides = pd.read_csv(landslide_csv_path)
    print(f"[+] Total landslide records loaded: {len(df_landslides)}")
    
    # Filter for India -> Himachal Pradesh
    hp_landslides = df_landslides[
        (df_landslides['country_code'] == 'IN') & 
        (df_landslides['admin_division_name'].str.contains('Himachal Pradesh', case=False, na=False))
    ]
    print(f"[+] Landslides recorded in Himachal Pradesh: {len(hp_landslides)}")
    
    # Filter for Mandi region bounding box (Lat: 31.2° - 32.3°, Lon: 76.4° - 77.5°)
    mandi_landslides = hp_landslides[
        (hp_landslides['latitude'].between(31.2, 32.3)) & 
        (hp_landslides['longitude'].between(76.4, 77.5))
    ]
    print(f"[+] Landslides in Mandi region: {len(mandi_landslides)}")
    
    # Save filtered subset
    output_csv = "mandi_landslides_processed.csv"
    mandi_landslides.to_csv(output_csv, index=False)
    print(f"[+] Mandi landslide subset saved to {output_csv}")
else:
    print(f"[-] Landslide CSV not found at {landslide_csv_path}")

# 3. Process ERA5 NetCDF Files from data/raw/ using h5py
era5_files = glob.glob(os.path.join(RAW_DIR, "*.nc"))
print(f"\n[*] Found {len(era5_files)} ERA5 NetCDF files in {RAW_DIR}/")

era5_data_summary = {}

for fpath in era5_files:
    fname = os.path.basename(fpath)
    try:
        with h5py.File(fpath, 'r') as h5f:
            print(f"\n--- File: {fname} ---")
            print("Variables inside:", list(h5f.keys()))
            
            # Extract coordinates and primary variables
            latitudes = h5f['latitude'][:] if 'latitude' in h5f else None
            longitudes = h5f['longitude'][:] if 'longitude' in h5f else None
            
            # Extracting variable arrays
            for key in h5f.keys():
                if key not in ['latitude', 'longitude', 'valid_time']:
                    data_array = h5f[key][:]
                    print(f"  -> Variable '{key}' shape: {data_array.shape}")
                    era5_data_summary[fname] = {
                        'variable': key,
                        'shape': data_array.shape,
                        'lat_range': (latitudes.min(), latitudes.max()) if latitudes is not None else None,
                        'lon_range': (longitudes.min(), longitudes.max()) if longitudes is not None else None
                    }
    except Exception as e:
        print(f"[-] Error reading {fname}: {e}")

print("\n[*] Data processing setup completed successfully for data/raw/ directory!")