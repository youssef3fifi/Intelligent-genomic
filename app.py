import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Load merged model
merged = joblib.load("merged_model.pkl")

app = Flask(__name__)

# Columns required by the model
FEATURES = ["Chromosome", "Gene", "Variant_Type", "CLNSIG", "Risk_Prob", "Risk_Level"]

@app.route("/")
def home():
    return jsonify({"message": "Genomic AI Model API is running"})


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "CSV file is required"}), 400

    file = request.files["file"]

    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify({"error": "Invalid CSV format"}), 400

    results = []

    for idx, row in df.iterrows():
        try:
            input_dict = {
                "Chromosome": str(row["Chromosome"]),
                "Gene": str(row["Gene"]),
                "Variant_Type": str(row["Variant_Type"]),
                "CLNSIG": str(row["CLNSIG"]),
                "Risk_Prob": float(row["Risk_Prob"]),
                "Risk_Level": str(row["Risk_Level"])
            }

            disease, treatment = merged.predict(input_dict)

            results.append({
                "ID": row.get("ID", f"row_{idx}"),
                "Disease": disease,
                "Treatment": treatment
            })

        except Exception as e:
            results.append({
                "ID": row.get("ID", f"row_{idx}"),
                "error": str(e)
            })

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
