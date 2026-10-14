import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load existing dataset
X = np.load("processed_data/convlstm_patches/X_patches.npy")
Y = np.load("processed_data/convlstm_patches/Y_patches.npy")

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Load the original ConvLSTM model
model = load_model("model.keras", compile=False)

# Balanced focal loss for rare fire pixels
loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True,
    alpha=0.90,
    gamma=2.0,
    from_logits=False
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss_fn
)

# Save the best model separately
checkpoint = ModelCheckpoint(
    "convlstm_focal_best.keras",
    monitor="loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

print("Starting ConvLSTM focal-loss training...")

history = model.fit(
    X,
    Y,
    epochs=40,
    batch_size=2,
    shuffle=True,
    callbacks=[checkpoint, early_stop]
)

# Save under a new filename; keep previous models untouched
model.save("convlstm_focal_trained.keras")

print("Training completed.")
print("New model: convlstm_focal_trained.keras")
print("Best checkpoint: convlstm_focal_best.keras")