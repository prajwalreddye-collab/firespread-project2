import os
import rasterio
import numpy as np

# Folder containing VIIRS_Day satellite images
image_folder = "dataset/images"

# Get all .tif files
image_files = sorted(
    [file for file in os.listdir(image_folder) if file.endswith(".tif")]
)

print("Number of satellite images:", len(image_files))
print()

# Read each image and display its shape
for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)

    with rasterio.open(file_path) as src:
        data = src.read()

        print("File:", file_name)
        print("Shape:", data.shape)
        print()

import os
import rasterio
import numpy as np

# Folder containing VIIRS_Day satellite images
image_folder = "dataset/images"

# Get all .tif files
image_files = sorted(
    [file for file in os.listdir(image_folder) if file.endswith(".tif")]
)

print("Number of satellite images:", len(image_files))
print()

# Read each image and display its shape
for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)

    with rasterio.open(file_path) as src:
        data = src.read()

        print("File:", file_name)
        print("Shape:", data.shape)
        print()


# Folder containing FirePred files
mask_folder = "dataset/masks"

# Get all .tif files
mask_files = sorted(
    [file for file in os.listdir(mask_folder) if file.endswith(".tif")]
)

print("Number of FirePred files:", len(mask_files))
print()

# Read each FirePred file and display its shape
for file_name in mask_files:
    file_path = os.path.join(mask_folder, file_name)

    with rasterio.open(file_path) as src:
        data = src.read()

        print("File:", file_name)
        print("Shape:", data.shape)
        print()

# Inspect metadata and values of one VIIRS_Day file

viirs_file = os.path.join(
    image_folder,
    "2017-04-23_VIIRS_Day.tif"
)

with rasterio.open(viirs_file) as src:
    data = src.read()

    print("\n--- VIIRS_Day DETAILS ---")
    print("CRS:", src.crs)
    print("Number of bands:", src.count)
    print("Data type:", src.dtypes)

    for band in range(src.count):
        print(
            f"Band {band + 1}: "
            f"Min = {np.nanmin(data[band])}, "
            f"Max = {np.nanmax(data[band])}"
        )


# Inspect metadata and values of one FirePred file

firepred_file = os.path.join(
    mask_folder,
    "2017-04-23_FirePred.tif"
)

with rasterio.open(firepred_file) as src:
    data = src.read()

    print("\n--- FIREPRED DETAILS ---")
    print("CRS:", src.crs)
    print("Number of bands:", src.count)
    print("Data type:", src.dtypes)

    for band in range(src.count):
        print(
            f"Band {band + 1}: "
            f"Min = {np.nanmin(data[band])}, "
            f"Max = {np.nanmax(data[band])}"
        )

print("\n--- VIIRS NaN CHECK ---")

with rasterio.open(viirs_file) as src:
    data = src.read()

    for band in range(src.count):
        total_pixels = data[band].size
        nan_pixels = np.isnan(data[band]).sum()
        valid_pixels = total_pixels - nan_pixels

        print(
            f"Band {band + 1}: "
            f"Valid = {valid_pixels}, "
            f"NaN = {nan_pixels}"
        )

print("\n--- VIIRS NaN CHECK FOR ALL DATES ---")

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)

    with rasterio.open(file_path) as src:
        data = src.read()

        print(f"\nFile: {file_name}")

        for band in range(src.count):
            total_pixels = data[band].size
            nan_pixels = np.isnan(data[band]).sum()
            valid_pixels = total_pixels - nan_pixels

            print(
                f"Band {band + 1}: "
                f"Valid = {valid_pixels}, "
                f"NaN = {nan_pixels}"
            )

print("\n--- CLEANING VIIRS DATA ---")

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)

    with rasterio.open(file_path) as src:
        data = src.read()

        # Keep only Bands 1–6
        data = data[:6]

        # Replace NaN values with 0
        data = np.nan_to_num(data, nan=0.0)

        print(f"\nFile: {file_name}")
        print("Cleaned shape:", data.shape)
        print("NaN values remaining:", np.isnan(data).sum())

print("\n--- NORMALIZING VIIRS DATA ---")

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)

    with rasterio.open(file_path) as src:
        data = src.read()

        # Keep Bands 1–6
        data = data[:6]

        # Replace NaN values with 0
        data = np.nan_to_num(data, nan=0.0)

        # Normalize each band
        normalized_data = np.zeros_like(data)

        for band in range(data.shape[0]):
            band_min = data[band].min()
            band_max = data[band].max()

            if band_max != band_min:
                normalized_data[band] = (
                    (data[band] - band_min) /
                    (band_max - band_min)
                )

        print(f"\nFile: {file_name}")
        print("Normalized shape:", normalized_data.shape)
        print(
            "Overall Min:",
            normalized_data.min(),
            "Overall Max:",
            normalized_data.max()
        )

import cv2

print("\n--- RESIZING VIIRS DATA ---")

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)

    with rasterio.open(file_path) as src:
        data = src.read()[:6]

        # Replace NaN values with 0
        data = np.nan_to_num(data, nan=0.0)

        # Normalize each band
        normalized_data = np.zeros_like(data)

        for band in range(data.shape[0]):
            band_min = data[band].min()
            band_max = data[band].max()

            if band_max != band_min:
                normalized_data[band] = (
                    (data[band] - band_min)
                    / (band_max - band_min)
                )

        # Resize each band to 224 × 224
        resized_data = np.zeros((6, 224, 224), dtype=np.float32)

        for band in range(6):
            resized_data[band] = cv2.resize(
                normalized_data[band],
                (224, 224)
            )

        print(f"\nFile: {file_name}")
        print("Original shape:", data.shape)
        print("Resized shape:", resized_data.shape)
        print(
            "Min:", resized_data.min(),
            "Max:", resized_data.max()
        )

# Create folder for processed data
processed_folder = "processed_data"
os.makedirs(processed_folder, exist_ok=True)

print("\n--- SAVING PROCESSED DATA ---")

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)

    with rasterio.open(file_path) as src:
        # Read only Bands 1–6
        data = src.read()[:6]

        # Replace NaN values with 0
        data = np.nan_to_num(data, nan=0.0)

        # Normalize each band
        normalized_data = np.zeros_like(data)

        for band in range(data.shape[0]):
            band_min = data[band].min()
            band_max = data[band].max()

            if band_max != band_min:
                normalized_data[band] = (
                    (data[band] - band_min)
                    / (band_max - band_min)
                )

        # Resize to 224 × 224
        resized_data = np.zeros((6, 224, 224), dtype=np.float32)

        for band in range(6):
            resized_data[band] = cv2.resize(
                normalized_data[band],
                (224, 224)
            )

        # Create output file name
        output_name = file_name.replace("_VIIRS_Day.tif", ".npy")
        output_path = os.path.join(processed_folder, output_name)

        # Save processed data
        np.save(output_path, resized_data)

        print("Saved:", output_path)
        print("Shape:", resized_data.shape)

print("\n--- VERIFYING SAVED DATA ---")

test_file = os.path.join(
    processed_folder,
    "2017-04-23.npy"
)

loaded_data = np.load(test_file)

print("Loaded file:", test_file)
print("Loaded shape:", loaded_data.shape)
print("Min:", loaded_data.min())
print("Max:", loaded_data.max())

print("\n--- FIREPRED BAND INFORMATION ---")

firepred_file = os.path.join(
    mask_folder,
    "2017-04-23_FirePred.tif"
)

with rasterio.open(firepred_file) as src:
    print("File:", firepred_file)
    print("Number of bands:", src.count)
    print("Image size:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("NoData value:", src.nodata)

    print("\nBand descriptions:")

    for band in range(src.count):
        print(
            f"Band {band + 1}:",
            src.descriptions[band]
        )

    print("\nDataset tags:")
    print(src.tags())