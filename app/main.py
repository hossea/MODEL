from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the models and scaler
with open('models/lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

with open('models/rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['features']])
    features_scaled = scaler.transform(features)
    
    lr_prediction = lr_model.predict(features_scaled)
    rf_prediction = rf_model.predict(features_scaled)
    
    response = {
        'lr_prediction': lr_prediction[0],
        'rf_prediction': rf_prediction[0]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
