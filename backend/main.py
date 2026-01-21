import os
import sys

# Fix Unicode encoding issues
if sys.platform == 'win32':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that cause issues

import warnings
import numpy as np

# Suppress NumPy dtype size warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# Check NumPy version for compatibility
if np.__version__.startswith('2'):
    raise ImportError("NumPy version 2.x detected, which is incompatible with TensorFlow. Please downgrade to NumPy <2.0 (e.g., 1.26.4) using 'pip install numpy==1.26.4' in your environment.")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- Load the trained model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_augmented.keras")
model = None
model_loaded = False

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please ensure 'best_model_augmented.keras' is in the backend directory.")
    model = load_model(MODEL_PATH)
    model_loaded = True
    print(f"[INFO] Model loaded successfully from '{MODEL_PATH}' ✅")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")
    print("[WARNING] Backend will start but prediction endpoint will not work until model is fixed.")
    model_loaded = False

# --- Automatically detect input size ---
input_shape = (150, 150, 3)  # Default fallback
if model_loaded:
    input_shape = model.input_shape[1:]  # (height, width, channels)
    print(f"[INFO] Model expects input shape: {input_shape}")

# --- Class labels (adjust if needed) ---
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Prediction function ---
def predict_tumor(img_path):
    # Load and preprocess the image properly
    img = image.load_img(img_path, target_size=input_shape[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Predict
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    # Display results
    print("\n--- Prediction Result ---")
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence Score: {confidence:.4f}")
    print(f"All Class Probabilities: {dict(zip(CLASS_NAMES, preds[0]))}")

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])

@app.route("/status", methods=["GET"])
def status():
    try:
        # Try a dummy prediction to check model health
        if model_loaded:
            print(f"[DEBUG] Attempting dummy prediction with input shape: {input_shape}")
            dummy = np.zeros((1, *input_shape))
            print(f"[DEBUG] Dummy array shape: {dummy.shape}")
            _ = model.predict(dummy, verbose=0)
            print("[DEBUG] Dummy prediction successful")
            return jsonify({"status": "ok", "model_loaded": True, "input_shape": input_shape}), 200
        else:
            return jsonify({"status": "partial", "model_loaded": False, "message": "Model could not be loaded", "input_shape": input_shape}), 200
    except Exception as e:
        print(f"[ERROR] Status check failed: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "model_loaded": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Please check server status and fix the model file."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join("/tmp", filename)
        file.save(filepath)
        try:
            # Use the same logic as predict_tumor
            img = image.load_img(filepath, target_size=input_shape[:2])
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            preds = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = float(np.max(preds))
            all_probs = {cls: float(prob) for cls, prob in zip(CLASS_NAMES, preds[0])}
            os.remove(filepath)
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_class_probabilities": all_probs
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Unknown error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
