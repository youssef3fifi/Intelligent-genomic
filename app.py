import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from model_class import MergedModel  # <--- السطر ده مهم

# 1. تهيئة تطبيق فلاسك
app = Flask(__name__)

# 2. تحميل المودل
print("Loading model...")
try:
    model = joblib.load("merged_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ... باقي الكود بتاع الـ endpoints زي ما هو ...
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model could not be loaded"}), 500
        
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        disease, treatment = model.predict(input_data)

        return jsonify({
            "predicted_disease": disease,
            "predicted_treatment": treatment
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return "Genomic Model API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)