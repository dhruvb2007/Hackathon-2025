from flask import Blueprint, request, jsonify
import numpy as np
from Model.model_loader import load_model  # Import model loader

predict_bp = Blueprint('predict', __name__)

# Define max values for normalization
max_values = {
    "Gujarati_PAT": 20, "Gujarati_SAT": 80,
    "Mathematics_PAT": 20, "Mathematics_SAT": 80,
    "Science_PAT": 20, "Science_SAT": 80,
    "SocialScience_PAT": 20, "SocialScience_SAT": 80,
    "English_PAT": 20, "English_SAT": 80,
    "Hindi_PAT": 20, "Hindi_SAT": 80,
    "Attendance": 100, "Assessment": 100
}

@predict_bp.route('/predict', methods=['GET'])
@cross_origin()
def predict():
    try:
        model = load_model()
        input_data = {key: float(request.args.get(key)) for key in max_values}
        input_data_normalized = [input_data[key] / max_values[key] for key in max_values]

        input_data_reshaped = np.array(input_data_normalized).reshape(1, -1)
        predicted_score = model.predict(input_data_reshaped)[0] * 100

        return jsonify({
            "Predicted_Final_Score": round(predicted_score, 2),
            "Input_Data": input_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
