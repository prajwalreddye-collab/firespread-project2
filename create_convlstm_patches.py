import numpy as np
import os

INPUT_DIR = "processed_data"
MASK_DIR = "processed_data/fire_masks"
OUTPUT_DIR = "processed_data/convlstm_patches"

PATCH_SIZE = 112
IMAGE_SIZE = 224
TIME_STEPS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

dates = [
    "2017-04-23",
    "2017-04-24",
    "2017-04-25",
    "2017-04-26",
    "2017-04-27"
]


# ----------------------------------------
# Resize binary mask from 595x595 to 224x224
# using nearest-neighbor sampling
# ----------------------------------------
def resize_mask(mask, new_height, new_width):

    old_height, old_width = mask.shape

    row_indices = (
        np.linspace(0, old_height - 1, new_height)
        .astype(int)
    )

    col_indices = (
        np.linspace(0, old_width - 1, new_width)
        .astype(int)
    )

    resized = mask[
        row_indices[:, None],
        col_indices[None, :]
    ]

    return resized


# ----------------------------------------
# Load satellite images
# ----------------------------------------

images = []

print("Loading satellite images...")

for date in dates:

    file_path = os.path.join(
        INPUT_DIR,
        date + ".npy"
    )

    image = np.load(file_path)

    print(
        date,
        "original processed shape:",
        image.shape
    )

    # (6,224,224) -> (224,224,6)
    image = np.transpose(
        image,
        (1, 2, 0)
    )

    images.append(image)


# ----------------------------------------
# Load and resize fire masks
# ----------------------------------------

masks = []

print()
print("Loading and resizing fire masks...")

for date in dates:

    file_path = os.path.join(
        MASK_DIR,
        date + "_fire_mask.npy"
    )

    mask = np.load(file_path)

    print(
        date,
        "original mask shape:",
        mask.shape
    )

    # Convert to binary
    mask = (mask > 0).astype(np.uint8)

    # 595x595 -> 224x224
    mask = resize_mask(
        mask,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    print(
        date,
        "resized mask shape:",
        mask.shape,
        "fire pixels:",
        np.sum(mask)
    )

    masks.append(mask)


# ----------------------------------------
# Create ConvLSTM sequences
# ----------------------------------------

X_patches = []
Y_patches = []

sample_number = 0

print()
print("Creating ConvLSTM patch sequences...")


for start in range(
    len(dates) - TIME_STEPS
):

    input_images = images[
        start:start + TIME_STEPS
    ]

    target_mask = masks[
        start + TIME_STEPS
    ]

    print()

    print(
        f"Sequence {sample_number + 1}: "
        f"{dates[start]} + "
        f"{dates[start + 1]} + "
        f"{dates[start + 2]}"
        f" -> {dates[start + TIME_STEPS]}"
    )

    # ----------------------------------------
    # Extract 112x112 patches
    # ----------------------------------------

    for row in range(
        0,
        IMAGE_SIZE,
        PATCH_SIZE
    ):

        for col in range(
            0,
            IMAGE_SIZE,
            PATCH_SIZE
        ):

            sequence_patch = []

            for t in range(TIME_STEPS):

                patch = input_images[t][
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE,
                    :
                ]

                sequence_patch.append(
                    patch
                )

            sequence_patch = np.array(
                sequence_patch,
                dtype=np.float32
            )

            # Target patch from the correctly resized mask
            target_patch = target_mask[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            target_patch = np.expand_dims(
                target_patch,
                axis=-1
            )

            X_patches.append(
                sequence_patch
            )

            Y_patches.append(
                target_patch
            )

    sample_number += 1


# ----------------------------------------
# Convert to NumPy arrays
# ----------------------------------------

X_patches = np.array(
    X_patches,
    dtype=np.float32
)

Y_patches = np.array(
    Y_patches,
    dtype=np.float32
)


# ----------------------------------------
# Display final shapes
# ----------------------------------------

print()
print("========================================")
print("CORRECTED CONVLSTM PATCH DATASET")
print("========================================")

print(
    "X shape:",
    X_patches.shape
)

print(
    "Y shape:",
    Y_patches.shape
)


# ----------------------------------------
# Save
# ----------------------------------------

X_file = os.path.join(
    OUTPUT_DIR,
    "X_patches.npy"
)

Y_file = os.path.join(
    OUTPUT_DIR,
    "Y_patches.npy"
)

np.save(
    X_file,
    X_patches
)

np.save(
    Y_file,
    Y_patches
)


print()
print("Saved:")
print(X_file)
print(Y_file)

print()
print("========================================")
print("PATCH CREATION COMPLETED")
print("========================================")