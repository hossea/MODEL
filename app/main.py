from flask import Flask, request, jsonify
import joblib
import pandas as pd
from utils import load_models_and_scaler, preprocess_user_input, predict_price

# Paths to the models, scaler, and columns
MODEL_PATH_LR = 'C:/Users/Windows 10 Pro/Desktop/housedata/models/lr_model.pkl'
SCALER_PATH = 'C:/Users/Windows 10 Pro/Desktop/housedata/models/scaler.pkl'
COLUMNS_PATH = 'C:/Users/Windows 10 Pro/Desktop/housedata/models/X_columns.pkl'

# Initialize Flask application
app = Flask(__name__)

# Load models, scaler, and column names
lr_model, scaler, X_columns = load_models_and_scaler(MODEL_PATH_LR, SCALER_PATH, COLUMNS_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        features = data['features']

        # Preprocess input features
        scaled_features = preprocess_user_input(features, X_columns, scaler)

        # Predict using models
        predicted_price_lr = predict_price(scaled_features, lr_model)

        return jsonify({
            'Linear Regression Prediction': predicted_price_lr,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
