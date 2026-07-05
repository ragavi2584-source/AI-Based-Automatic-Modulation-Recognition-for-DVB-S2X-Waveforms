"""
app.py
Flask Web Application for
AI-Based Automatic Modulation Recognition
"""

import os
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename

from predict import (
    load_signal,
    predict_with_time
)

from visualization import (
    generate_all_plots,
    clear_plots
)

# ---------------------------------------
# Flask Configuration
# ---------------------------------------

app = Flask(__name__)

app.secret_key = "hackathon_secret_key"

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "npy"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ---------------------------------------
# Helper Function
# ---------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# ---------------------------------------
# Home Page
# ---------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )

# ---------------------------------------
# Upload Page
# ---------------------------------------

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "signal" not in request.files:

        flash("No file selected.")

        return redirect("/")

    file = request.files["signal"]

    if file.filename == "":

        flash("Choose a .npy file.")

        return redirect("/")

    if not allowed_file(file.filename):

        flash("Unsupported file type.")

        return redirect("/")

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    return redirect(
        url_for(
            "predict",
            filename=filename
        )
    )
# ---------------------------------------
# Prediction Route
# ---------------------------------------

@app.route("/predict/<filename>")
def predict(filename):

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        # Remove previous plots
        clear_plots()

        # Load uploaded signal
        signal = load_signal(filepath)

        # Predict modulation
        result = predict_with_time(signal)

        # Create confidence vector
        confidence_vector = [0.0, 0.0, 0.0]

        confidence_vector[
            result["class_index"]
        ] = result["confidence"]

        # Generate plots
        generate_all_plots(
            signal=signal,
            prediction=result["modulation"],
            confidence=confidence_vector
        )

        return render_template(

            "result.html",

            modulation=result["modulation"],

            confidence=round(
                result["confidence"] * 100,
                2
            ),

            prediction_time=result["prediction_time"],

            constellation="plots/constellation.png",

            waveform="plots/waveform.png",

            fft="plots/fft.png",

            spectrogram="plots/spectrogram.png",

            confidence_plot="plots/confidence.png"

        )

    except Exception as e:

        return render_template(

            "result.html",

            error=str(e)

        )
# ---------------------------------------
# About Page (Optional)
# ---------------------------------------

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ---------------------------------------
# Health Check
# ---------------------------------------

@app.route("/health")
def health():

    return {
        "status": "Running",
        "application": "AI-Based Automatic Modulation Recognition",
        "framework": "Flask"
    }


# ---------------------------------------
# 404 Error
# ---------------------------------------

@app.errorhandler(404)
def page_not_found(error):

    return (
        render_template(
            "result.html",
            error="404 - Page Not Found"
        ),
        404
    )


# ---------------------------------------
# 500 Error
# ---------------------------------------

@app.errorhandler(500)
def internal_error(error):

    return (
        render_template(
            "result.html",
            error="Internal Server Error"
        ),
        500
    )


# ---------------------------------------
# Cleanup Uploads (Optional)
# ---------------------------------------

def cleanup_uploads():

    folder = app.config["UPLOAD_FOLDER"]

    if not os.path.exists(folder):
        return

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            os.remove(path)
        except Exception:
            pass


# ---------------------------------------
# Run Flask App
# ---------------------------------------

if __name__ == "__main__":

    cleanup_uploads()

    print("=" * 50)
    print(" AI-Based Automatic Modulation Recognition")
    print(" DVB-S2X Waveforms")
    print("=" * 50)
    print("Server starting...")
    print("Open: http://127.0.0.1:5000")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )