import os
import numpy as np
import cv2

dates = [
    "2017-04-23",
    "2017-04-24",
    "2017-04-25",
    "2017-04-26",
    "2017-04-27"
]

X = []
Y = []

print("--- CREATING TIME-SERIES SEQUENCES ---")

for i in range(len(dates) - 3):
    input_sequence = []

    # Use 3 days as input
    for j in range(3):
        date = dates[i + j]

        image_path = f"processed_data/{date}.npy"
        image = np.load(image_path)

        input_sequence.append(image)

    # Next day's fire mask is the target
    target_date = dates[i + 3]
    mask_path = (
        f"processed_data/fire_masks/"
        f"{target_date}_fire_mask.npy"
    )

    mask = np.load(mask_path)

    # Resize mask using nearest-neighbor interpolation
    mask = cv2.resize(
        mask,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
    )

    X.append(np.stack(input_sequence))
    Y.append(mask)

X = np.array(X, dtype=np.float32)
X = np.transpose(X, (0, 1, 3, 4, 2))  # (num_sequences, num_days, bands, height, width)
Y = np.array(Y, dtype=np.float32)

Y = Y[..., np.newaxis]

os.makedirs("processed_data/sequences", exist_ok=True)

np.save("processed_data/sequences/X.npy", X)
np.save("processed_data/sequences/Y.npy", Y)

print("Input shape:", X.shape)
print("Target shape:", Y.shape)
print("Saved X.npy and Y.npy")