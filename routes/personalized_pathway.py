from flask import Blueprint, request, jsonify
import google.generativeai as genai
import joblib
import numpy as np
import json


pathway_bp = Blueprint('personalized_pathway', __name__)

# Directly pass your Gemini API key here
genai.configure(api_key="AIzaSyAmNRvyNqBdFtotLiD-i6rmVrXf1Gg9Rc4")

# Load the trained model
model = joblib.load("student_performance_model.pkl")

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

def preprocess_input(input_data):
    """Normalize input values and reshape for prediction."""
    input_data_normalized = [input_data[key] / max_values[key] for key in max_values]
    return np.array(input_data_normalized).reshape(1, -1)

def generate_learning_path(predicted_score, input_data):
    """Generate a personalized learning pathway using Gemini API."""
    prompt = f"""
    A student has the following academic performance:
    {input_data}

    The predicted final performance score is {predicted_score}/100.
    
    Based on this, suggest a **personalized learning pathway** focusing on:
    - **Strengths**: Subjects where the student is doing well.
    - **Weaknesses**: Subjects where improvement is needed.
    - **Study Plan**: Weekly plan to improve performance.
    - **Resources**: Recommended books, online courses, or study strategies.
    
    Provide the response in a structured JSON format.
    """

    model_gemini = genai.GenerativeModel("gemini-2.0-flash")  # Use the correct model name
    response = model_gemini.generate_content(prompt)

    # Clean the response text (remove code block markers)
    if response and hasattr(response, "text"):
        # Remove the '```json' and closing '```' markers
        cleaned_response = response.text.replace('```json\n', '').replace('\n```', '')

        # Try to load the cleaned response into a JSON object
        try:
            json_response = json.loads(cleaned_response)  # Parse the cleaned response as JSON
            return jsonify({
                "Personalized_Pathway": json_response,
                "Predicted_Final_Score": round(predicted_score, 2)
            })
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format in Gemini API response"}), 500

    return jsonify({"error": "No response from Gemini API"}), 500

@pathway_bp.route('/personalized_pathway', methods=['GET'])
def personalized_pathway():
    try:
        # Get input data
        input_data = {key: float(request.args.get(key)) for key in max_values}

        # Preprocess input
        input_data_reshaped = preprocess_input(input_data)

        # Predict score
        predicted_score = model.predict(input_data_reshaped)[0] * 100

        # Get personalized learning path from Gemini API
        return generate_learning_path(predicted_score, input_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
