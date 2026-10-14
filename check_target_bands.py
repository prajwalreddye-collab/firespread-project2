import numpy as np

path = "processed_data/fire_masks/2017-04-23_fire_mask.npy"

mask = np.load(path)

print("--- VERIFYING FIRE MASK ---")
print("Shape:", mask.shape)
print("Unique values:", np.unique(mask))
print("Fire pixels:", int(mask.sum()))
print("Non-fire pixels:", int((mask == 0).sum()))