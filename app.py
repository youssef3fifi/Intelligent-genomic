import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# --- !! مهم جداً: انسخ تعريف الكلاس بتاعك هنا ---
# The class definition must be present for joblib.load() to work
class MergedModel:
    def __init__(self, disease_model, treatment_model, enc_disease, enc_treatment, le_disease, le_treatment, features):
        self.disease_model = disease_model
        self.treatment_model = treatment_model
        self.enc_disease = enc_disease
        self.enc_treatment = enc_treatment
        self.le_disease = le_disease
        self.le_treatment = le_treatment
        self.features = features

    def _encode(self, input_dict, enc):
        inp = input_dict.copy()
        for col in ["Chromosome", "Gene", "Variant_Type", "CLNSIG", "Risk_Level"]:
            if col not in inp:
                raise ValueError(f"Missing {col}")
            val = str(inp[col])
            le = enc[col]
            if val not in le.classes_:
                # في التطبيق الفعلي، ممكن تتعامل مع القيمة دي بشكل مختلف
                # لكن حالياً هنرمي خطأ زي الكود بتاعك
                raise ValueError(f"Unseen value {val} for {col}")
            inp[col] = le.transform([val])[0]
        
        if "Risk_Prob" not in inp:
            raise ValueError("Missing Risk_Prob")
        inp["Risk_Prob"] = float(inp["Risk_Prob"])
        
        # التأكد من ترتيب الأعمدة
        return pd.DataFrame([inp])[self.features]

    def predict(self, input_dict):
        Xd = self._encode(input_dict, self.enc_disease)
        Xt = self._encode(input_dict, self.enc_treatment)
        d_enc = self.disease_model.predict(Xd)[0]
        t_enc = self.treatment_model.predict(Xt)[0]
        return (self.le_disease.inverse_transform([d_enc])[0],
                self.le_treatment.inverse_transform([t_enc])[0])

# --- نهاية تعريف الكلاس ---

# 1. تهيئة تطبيق فلاسك
app = Flask(__name__)

# 2. تحميل المودل (ملف المودل لازم يكون في نفس الفولدر)
print("Loading model...")
try:
    model = joblib.load("merged_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle error appropriately

# 3. تعريف نقطة النهاية (endpoint) للتوقع
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model could not be loaded"}), 500
        
    try:
        # 1. هات الداتا الـ JSON من الـ request
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # 2. استخدم المودل بتاعك للتوقع
        disease, treatment = model.predict(input_data)

        # 3. رجّع النتيجة كـ JSON
        return jsonify({
            "predicted_disease": disease,
            "predicted_treatment": treatment
        })

    except ValueError as ve:
        # ده هيمسك الأخطاء اللي جاية من _encode زي "Unseen value"
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# (اختياري) endpoint رئيسي عشان تتأكد إن السيرفر شغال
@app.route('/', methods=['GET'])
def home():
    return "Genomic Model API is running!"

# الكود ده عشان لو حبيت تشغله محلياً (local)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
