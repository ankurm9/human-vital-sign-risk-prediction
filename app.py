from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
model = load_model('model/ANN.h5')  # Load your trained model

# Dummy scaler - replace with your actual trained scaler
scaler = StandardScaler()

def preprocess_input(input_data):
    return scaler.fit_transform([input_data])[0]  # Replace fit_transform with transform when using real scaler

@app.route('/')
def index():
    # Label, name attribute, and default value
    inputs = [
        ("Heart Rate", "heart_rate", 60),
        ("Respiratory Rate", "resp_rate", 12),
        ("Body Temperature (°C)", "body_temp", 36.0),
        ("Oxygen Saturation (%)", "oxygen_sat", 95.0),
        ("Systolic BP", "systolic_bp", 110),
        ("Diastolic BP", "diastolic_bp", 70),
        ("Age", "age", 18),
        ("Gender (0=Male, 1=Female)", "gender", 0),
        ("Weight (kg)", "weight", 50.0),
        ("Height (m)", "height", 1.5),
        ("HRV", "hrv", 0.08),
        ("Pulse Pressure", "pulse_pressure", 30),
        ("BMI", "bmi", 20.3),
        ("MAP", "map", 88.7),
    ]
    return render_template('index.html', inputs=inputs)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            user_input = [float(request.form[key]) for key in request.form]
            processed_input = preprocess_input(user_input)
            prediction = model.predict(np.array([processed_input]))[0][0]
            risk_label = "High Risk" if prediction >= 0.5 else "Low Risk"
            return render_template('result.html', prediction=round(prediction, 2), risk_label=risk_label)
        except Exception as e:
            return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
