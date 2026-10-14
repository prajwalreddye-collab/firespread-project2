import numpy as np
import matplotlib.pyplot as plt

path = "processed_data/fire_masks/2017-04-23_fire_mask.npy"

mask = np.load(path)

plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap="hot")
plt.title("Active Fire Mask - 2017-04-23")
plt.colorbar(label="Fire Mask (0 or 1)")
plt.axis("off")
plt.show()