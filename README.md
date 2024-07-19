# House Price Prediction Project

This project uses a trained machine learning model to predict house prices.

## Requirements

- Python 3.7+
- Flask
- joblib
- pandas
- numpy
- scikit-learn

## Setup

1. Clone the repository or unzip the files to your local machine.
2. Navigate to the project directory.
3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Run the Flask app:
    ```sh
    python app.py
    ```
5. The API will be available at `http://localhost:5000`.

## Usage

To make a prediction, send a POST request to the `/predict` endpoint with a JSON payload containing the features. For example:

```json
{
    "feature1": "value1",
    "feature2": "value2"
   
}
