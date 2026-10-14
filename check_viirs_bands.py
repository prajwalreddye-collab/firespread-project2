import os
import numpy as np
import tifffile

image_folder = "dataset/images"

image_files = sorted(
    file for file in os.listdir(image_folder)
    if file.endswith(".tif")
)

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)
    data = tifffile.imread(file_path)

    print("\nFile:", file_name)
    print("Shape:", data.shape)

    for band in range(data.shape[2]):
        band_data = data[:, :, band]
        valid = np.isfinite(band_data)

        print(
            f"Band {band + 1}: "
            f"Valid = {valid.sum()}, "
            f"Missing = {(~valid).sum()}"
        )

        if valid.any():
            print(
                "  Min:", band_data[valid].min(),
                "Max:", band_data[valid].max()
            )