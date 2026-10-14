import numpy as np
from tensorflow.keras.models import load_model

# File paths
X_FILE = "processed_data/convlstm_patches/X_patches.npy"
Y_FILE = "processed_data/convlstm_patches/Y_patches.npy"
MODEL_FILE = "convlstm_trained.keras"

print("Loading prepared data...")
X = np.load(X_FILE)
Y = np.load(Y_FILE)

print("Input shape:", X.shape)
print("Actual mask shape:", Y.shape)

print("\nLoading trained ConvLSTM model...")
model = load_model(MODEL_FILE, compile=False)

print("\nGenerating predictions...")
probabilities = model.predict(X, verbose=1)

print("Prediction shape:", probabilities.shape)
print("Minimum probability:", float(probabilities.min()))
print("Maximum probability:", float(probabilities.max()))
print("Mean probability:", float(probabilities.mean()))

# Evaluate at the standard 0.5 threshold
threshold = 0.5
predicted = (probabilities >= threshold).astype(np.uint8)
actual = (Y >= 0.5).astype(np.uint8)

tp = int(np.sum((predicted == 1) & (actual == 1)))
fp = int(np.sum((predicted == 1) & (actual == 0)))
fn = int(np.sum((predicted == 0) & (actual == 1)))
tn = int(np.sum((predicted == 0) & (actual == 0)))

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

print("\n--- Evaluation Results ---")
print("Threshold:", threshold)
print("Actual fire pixels:", int(actual.sum()))
print("Predicted fire pixels:", int(predicted.sum()))
print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print("True negatives:", tn)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"Dice:      {dice:.4f}")

print("\nEvaluation finished.")