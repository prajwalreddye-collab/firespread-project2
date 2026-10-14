import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

MODEL_FILE = "convlstm_trained.keras"
X_FILE = "processed_data/convlstm_patches/X_patches.npy"
OUTPUT_DIR = "static"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading trained ConvLSTM model...")
model = load_model(MODEL_FILE, compile=False)
print("Model loaded successfully.")

X = np.load(X_FILE)
print("Input shape:", X.shape)

X_test = X[0:1]

print("Generating prediction...")
prediction = model.predict(X_test, verbose=0)
probability_map = prediction[0, :, :, 0]

print("Minimum probability:", float(np.min(probability_map)))
print("Maximum probability:", float(np.max(probability_map)))
print("Average probability:", float(np.mean(probability_map)))

# Save probability map as an image
output_path = os.path.join(OUTPUT_DIR, "prediction_map.png")

plt.figure(figsize=(8, 6))
plt.imshow(probability_map, cmap="hot", vmin=0, vmax=1)
plt.colorbar(label="Predicted fire probability")
plt.title("ConvLSTM Fire Probability Map")
plt.axis("off")
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()

print("Prediction image saved to:", output_path)
print("Prediction image is a model output, not a validated fire forecast.")