import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path to DEM
dem_path = Path(__file__).parent.parent / "data" / "ldem_85s_20m_float.lbl"

# Open DEM
dem = rasterio.open(dem_path)

# Read first band
elevation = dem.read(1)

# Print information
print("=" * 50)
print("DEM Information")
print("=" * 50)

print(f"Width        : {dem.width}")
print(f"Height       : {dem.height}")
print(f"Bands        : {dem.count}")
print(f"Data Type    : {dem.dtypes[0]}")
print(f"CRS          : {dem.crs}")
print(f"Transform    :")
print(dem.transform)

print("\nElevation Statistics")
print("---------------------")
print(f"Minimum : {np.nanmin(elevation)}")
print(f"Maximum : {np.nanmax(elevation)}")
print(f"Mean    : {np.nanmean(elevation)}")

# Display DEM
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap="gray")
plt.colorbar(label="Elevation")
plt.title("LOLA DEM")
plt.show()