import numpy as np

# -----------------------------------------
# FILES
# -----------------------------------------

PREDICTION_FILE = (
    "processed_data/convlstm_patches/predicted_fire.npy"
)

ACTUAL_MASK_FILE = (
    "processed_data/fire_masks/2017-04-26_fire_mask.npy"
)

# -----------------------------------------
# LOAD DATA
# -----------------------------------------

prediction = np.load(PREDICTION_FILE)

actual = np.load(ACTUAL_MASK_FILE)

# Remove batch/channel dimensions
prediction = prediction[0, :, :, 0]

# IMPORTANT:
# Our prediction is 112x112.
# The original mask is 595x595.
#
# The first patch corresponds to the
# top-left 112x112 region.

actual = actual[:112, :112]

# Convert to binary
prediction = prediction.astype(np.uint8)
actual = (actual > 0).astype(np.uint8)

# -----------------------------------------
# CALCULATE CONFUSION VALUES
# -----------------------------------------

true_positive = np.sum(
    (prediction == 1) & (actual == 1)
)

true_negative = np.sum(
    (prediction == 0) & (actual == 0)
)

false_positive = np.sum(
    (prediction == 1) & (actual == 0)
)

false_negative = np.sum(
    (prediction == 0) & (actual == 1)
)

# -----------------------------------------
# METRICS
# -----------------------------------------

precision = (
    true_positive /
    (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive /
    (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

iou = (
    true_positive /
    (true_positive + false_positive + false_negative)
    if (true_positive + false_positive + false_negative) > 0
    else 0
)

# -----------------------------------------
# DISPLAY RESULTS
# -----------------------------------------

print()
print("========================================")
print("CONVLSTM EVALUATION")
print("========================================")

print("Actual fire pixels:", np.sum(actual))
print("Predicted fire pixels:", np.sum(prediction))

print()
print("True Positive :", true_positive)
print("True Negative :", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)

print()
print("Precision:", precision)
print("Recall   :", recall)
print("IoU      :", iou)

print()
print("========================================")