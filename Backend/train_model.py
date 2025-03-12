import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../Data/gujarat_students_data.csv")

# Define maximum values for normalization
max_values = {
    "Gujarati_PAT": 20, "Gujarati_SAT": 80,
    "Mathematics_PAT": 20, "Mathematics_SAT": 80,
    "Science_PAT": 20, "Science_SAT": 80,
    "SocialScience_PAT": 20, "SocialScience_SAT": 80,
    "English_PAT": 20, "English_SAT": 80,
    "Hindi_PAT": 20, "Hindi_SAT": 80,
    "Attendance": 100, "Assessment": 100,
    "Final_Score": 100  # Target variable max
}

# Normalize the data
for col in max_values:
    if col in df.columns:
        df[col] = df[col] / max_values[col]

# Select Features & Target
feature_columns = [col for col in df.columns if col != "Final_Score"]
X = df[feature_columns]
y = df["Final_Score"]  # Using normalized Final Score

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save Model for Future Use
joblib.dump(model, "../student_performance_model.pkl")
print("✅ Model trained with normalization and saved successfully!")
