import os
import numpy as np
from flask import Flask, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_FILE = "convlstm_trained.keras"
X_FILE = "processed_data/convlstm_patches/X_patches.npy"


@app.route("/")
def dashboard():
    stats = {
        "minimum": "Unavailable",
        "maximum": "Unavailable",
        "average": "Unavailable",
        "status": "Prediction statistics unavailable"
    }

    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(X_FILE):
            model = load_model(MODEL_FILE, compile=False)
            X = np.load(X_FILE)
            prediction = model.predict(X[0:1], verbose=0)
            probability_map = prediction[0, :, :, 0]

            stats = {
                "minimum": f"{probability_map.min():.4f}",
                "maximum": f"{probability_map.max():.4f}",
                "average": f"{probability_map.mean():.4f}",
                "status": "Prediction generated"
            }
    except Exception as error:
        print("Could not calculate prediction statistics:", error)

    return render_template("dashboard.html", stats=stats)


if __name__ == "__main__":
    app.run(debug=True)