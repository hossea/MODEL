from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model_pipeline = joblib.load('linear_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure input_data has the same columns as the training data
        required_columns = model_pipeline.named_steps['preprocessor'].transformers_[0][2]

        for column in required_columns:
            if column not in input_data.columns:
                input_data[column] = np.nan

        input_data = input_data[required_columns]
        
        # Preprocess and predict
        input_data_processed = model_pipeline.named_steps['preprocessor'].transform(input_data)
        prediction = model_pipeline.named_steps['regressor'].predict(input_data_processed)
        
        # Return the prediction
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
