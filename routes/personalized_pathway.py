from flask import Blueprint, request, jsonify
from flask_cors import CORS, cross_origin
import google.generativeai as genai
import numpy as np
import json
from Model.model_loader import load_model  # Import model loader

pathway_bp = Blueprint('personalized_pathway', __name__)

# Configure Gemini API Key
genai.configure(api_key="AIzaSyAmNRvyNqBdFtotLiD-i6rmVrXf1Gg9Rc4")

max_values = {
    "Gujarati_PAT": 20, "Gujarati_SAT": 80,
    "Mathematics_PAT": 20, "Mathematics_SAT": 80,
    "Science_PAT": 20, "Science_SAT": 80,
    "SocialScience_PAT": 20, "SocialScience_SAT": 80,
    "English_PAT": 20, "English_SAT": 80,
    "Hindi_PAT": 20, "Hindi_SAT": 80,
    "Attendance": 100, "Assessment": 100
}

def preprocess_input(input_data):
    return np.array([input_data[key] / max_values[key] for key in max_values]).reshape(1, -1)

def generate_learning_path(predicted_score, input_data):
    prompt = f"""
    A student has the following academic performance:
    {input_data}

    The predicted final performance score is {predicted_score}/100.
    
    Suggest a **personalized learning pathway**:
    - **Strengths**
    - **Weaknesses**
    - **Study Plan**
    - **Resources**
    
    Provide the response in JSON format.
    """

    model_gemini = genai.GenerativeModel("gemini-2.0-flash")
    response = model_gemini.generate_content(prompt)

    if response and hasattr(response, "text"):
        try:
            json_response = json.loads(response.text.replace('```json\n', '').replace('\n```', ''))
            return jsonify({"Personalized_Pathway": json_response, "Predicted_Final_Score": round(predicted_score, 2)})
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format in Gemini API response"}), 500

    return jsonify({"error": "No response from Gemini API"}), 500

@pathway_bp.route('/personalized_pathway', methods=['GET'])
@cross_origin()
def personalized_pathway():
    try:
        model = load_model()
        input_data = {key: float(request.args.get(key)) for key in max_values}
        input_data_reshaped = preprocess_input(input_data)
        predicted_score = model.predict(input_data_reshaped)[0] * 100
        return generate_learning_path(predicted_score, input_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
