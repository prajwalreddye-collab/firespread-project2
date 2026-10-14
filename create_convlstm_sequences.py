import os
import numpy as np

# Folder containing the fire masks
MASK_FOLDER = "processed_data/fire_masks"

# Output folder for ConvLSTM sequences
OUTPUT_FOLDER = "processed_data/convlstm_sequences"

# Create output folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Dates in chronological order
dates = [
    "2017-04-23",
    "2017-04-24",
    "2017-04-25",
    "2017-04-26",
    "2017-04-27"
]

# Load all masks
masks = []

for date in dates:
    path = os.path.join(
        MASK_FOLDER,
        f"{date}_fire_mask.npy"
    )

    mask = np.load(path)

    print(f"Loaded: {date}")
    print(f"Shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")

    masks.append(mask)

# Convert list to NumPy array
masks = np.array(masks, dtype=np.float32)

print("\nAll masks combined")
print("Shape:", masks.shape)

# --------------------------------------------------
# Create 3-day input sequences
# --------------------------------------------------

X = []
Y = []

# 3 consecutive days -> next day
sequence_length = 3

for i in range(len(masks) - sequence_length):

    input_sequence = masks[i:i + sequence_length]
    target = masks[i + sequence_length]

    X.append(input_sequence)
    Y.append(target)

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# Add channel dimension
# X: (samples, time, height, width, channels)
X = np.expand_dims(X, axis=-1)

# Y: (samples, height, width, channels)
Y = np.expand_dims(Y, axis=-1)

print("\nConvLSTM sequence data")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Save
np.save(
    os.path.join(OUTPUT_FOLDER, "X.npy"),
    X
)

np.save(
    os.path.join(OUTPUT_FOLDER, "Y.npy"),
    Y
)

print("\nSaved successfully!")

print(
    "X:",
    os.path.join(OUTPUT_FOLDER, "X.npy")
)

print(
    "Y:",
    os.path.join(OUTPUT_FOLDER, "Y.npy")
)