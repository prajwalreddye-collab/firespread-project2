import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

# --------------------------------------------------
# CONVLSTM MODEL
# --------------------------------------------------

model = Sequential([
    ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="tanh",
        return_sequences=False,
        input_shape=(3, 112, 112, 6)
    ),

    BatchNormalization(),

    Conv2D(
        filters=1,
        kernel_size=(1, 1),
        activation="sigmoid",
        padding="same"
    )
])


# --------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# DISPLAY MODEL
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model.save("model.keras")

print("Model saved successfully")