import numpy as np
from flask import Flask, request, jsonify
import joblib

# Load model
model = joblib.load("student_performance_model.pkl")

# Max values for normalization
max_values = {
    "Gujarati_PAT": 20, "Gujarati_SAT": 80,
    "Mathematics_PAT": 20, "Mathematics_SAT": 80,
    "Science_PAT": 20, "Science_SAT": 80,
    "SocialScience_PAT": 20, "SocialScience_SAT": 80,
    "English_PAT": 20, "English_SAT": 80,
    "Hindi_PAT": 20, "Hindi_SAT": 80,
    "Attendance": 100, "Assessment": 100
}

app = Flask(__name__)

def get_performance_statement(score):
    if score >= 95:
        return "You're among the top performers! Keep up the outstanding work and aim for mastery in every subject."
    elif score >= 90:
        return "Fantastic performance! Stay consistent and challenge yourself to reach even greater heights."
    elif score >= 80:
        return "Great job! You have a strong grasp of the concepts. Keep pushing for excellence!"
    elif score >= 70:
        return "You're doing well! Focus on areas where you can improve to boost your score even higher."
    elif score >= 60:
        return "You have a good foundation, but some subjects may need extra attention. Keep practicing!"
    elif score >= 50:
        return "You're on the right path, but more effort is needed. Review weak areas and ask for help if needed."
    elif score >= 40:
        return "Your performance is concerning. Try to dedicate more time to studying and improve your understanding."
    elif score >= 30:
        return "You need significant improvement. Seek help, attend extra classes, and work on study habits."
    elif score >= 20:
        return "You're struggling a lot. Consider speaking with a mentor or teacher to find ways to improve."
    else:
        return "Immediate action is needed! Develop a strong study plan, seek guidance, and work hard to improve."

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get input data
        input_data = {key: float(request.args.get(key)) for key in max_values}

        # Normalize input
        input_data_normalized = [input_data[key] / max_values[key] for key in max_values]

        # Convert to 2D array
        input_data_reshaped = np.array(input_data_normalized).reshape(1, -1)

        # Predict score and scale back to 0-100
        predicted_score = model.predict(input_data_reshaped)[0] * 100

        # Get performance statement
        performance_statement = get_performance_statement(predicted_score)

        return jsonify({
            "Predicted_Final_Score": round(predicted_score, 2),
            "Performance_Statement": performance_statement,
            "Input_Data": input_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
