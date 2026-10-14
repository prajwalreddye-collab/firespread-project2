import numpy as np
import tifffile

path = "dataset/masks/2017-04-23_FirePred.tif"

data = tifffile.imread(path)

print("Full shape:", data.shape)

for band in range(data.shape[2]):
    band_data = data[:, :, band]
    valid = np.isfinite(band_data)

    print(f"\nBand {band + 1}")
    print("Valid pixels:", valid.sum())

    if valid.any():
        print("Min:", band_data[valid].min())
        print("Max:", band_data[valid].max())
        print("Unique values:", np.unique(band_data[valid])[:10])