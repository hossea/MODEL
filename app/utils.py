import pandas as pd
import joblib

def load_models_and_scaler(model_path_lr, scaler_path, columns_path):
    """
    Load the models, scaler, and column names.
    """
    lr_model = joblib.load(model_path_lr)
    scaler = joblib.load(scaler_path)
    X_columns = joblib.load(columns_path)
    return lr_model, scaler, X_columns

def preprocess_user_input(user_input, X_columns, scaler):
    """
    Preprocess the user input data to match the training data format.
    """
    features = pd.DataFrame([user_input])
    
    # Handle missing columns and one-hot encoding
    features = pd.get_dummies(features, drop_first=True)
    
    # Ensure all expected columns are present
    features = features.reindex(columns=X_columns, fill_value=0)
    
    # Scale features
    scaled_features = scaler.transform(features)
    return scaled_features

def predict_price(features, model):
    """
    Predict the price using the provided model and preprocessed features.
    """
    return model.predict(features)[0]
