from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and the scaler
model = joblib.load("C:/Users/Windows 10 Pro/Desktop/housedata/models/house_price_model.pkl")
scaler = joblib.load("C:/Users/Windows 10 Pro/Desktop/housedata/models/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Debugging: Print the received data
        print("Received data:", data)
        
        # Extract and transform features
        features = np.array([data['size(sqft)'], data['parking_lot'], data['rooms'], data['year_built']]).reshape(1, -1)
        
        # Debugging: Print the extracted features
        print("Extracted features:", features)
        
        features_scaled = scaler.transform(features)
        
        # Debugging: Print the scaled features
        print("Scaled features:", features_scaled)
        
        prediction = model.predict(features_scaled)
        
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        # Debugging: Print the exception message
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
