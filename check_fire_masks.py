import numpy as np
import os
import glob

folder = "processed_data/fire_masks"

files = sorted(glob.glob(os.path.join(folder, "*.npy")))

print("Number of masks:", len(files))

for file_path in files:
    mask = np.load(file_path)

    fire_pixels = np.sum(mask == 1)
    non_fire_pixels = np.sum(mask == 0)

    print("\nFile:", os.path.basename(file_path))
    print("Shape:", mask.shape)
    print("Fire pixels:", fire_pixels)
    print("Non-fire pixels:", non_fire_pixels)
    print("Unique values:", np.unique(mask))