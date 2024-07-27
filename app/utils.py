from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('C:/Users/Windows 10 Pro/Desktop/housedata/models/house_price_model.pkl')
scaler = joblib.load('C:/Users/Windows 10 Pro/Desktop/housedata/models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Extract features from the data
        features = [
            data['ZIP'], data['images_count'], data['SQUARE_FT'], 
            data['parking_lot'], data['rooms'], data['user_id'], 
            data['phone'], data['year_built'], data['Yr_sold'], data['sale_type']
        ]
        
        # Convert the features to a numpy array and reshape for a single sample
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make the prediction
        prediction = model.predict(scaled_features)
        
        # Return the prediction as JSON
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
