import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

def load_data(file_path="discretized.csv"):
    """
    Load the dataset from CSV
    
    Parameters:
    file_path (str): Path to the CSV dataset
    
    Returns:
    DataFrame: Loaded dataframe
    """
    return pd.read_csv(file_path)

def prepare_data(df, test_size=0.3, random_state=42):
    """
    Split the dataset into features and target, then into train/test sets
    
    Parameters:
    df (DataFrame): The input dataframe
    test_size (float): Proportion of data to use for testing (default: 0.3)
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    # Split into features and target
    X = df[['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    y = df['CLASS']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Save the train and test data as CSV for observation (optional)
    save_train_test_data(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test

def save_train_test_data(X_train, X_test, y_train, y_test):
    """
    Save train and test data as CSV files
    
    Parameters:
    X_train, X_test: Features for training and testing
    y_train, y_test: Target for training and testing
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save training data
    train_data = X_train.copy()
    train_data['CLASS'] = y_train
    train_data.to_csv("data/train_data.csv", index=False)
    
    # Save testing data
    test_data = X_test.copy()
    test_data['CLASS'] = y_test
    test_data.to_csv("data/test_data.csv", index=False)
    
    print("Train and test data saved to 'data' directory")

def prepare_scaled_data(X_train, X_test):
    """
    Scale the data using RobustScaler
    
    Parameters:
    X_train, X_test: Features for training and testing
    
    Returns:
    tuple: X_train_scaled, X_test_scaled, scaler
    """
    # Apply RobustScaler to scale the data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future predictions
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, 'models/robust_scaler.pkl')
    
    # Save the scaled data as CSV for observation (optional)
    save_scaled_data(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    
    return X_train_scaled, X_test_scaled, scaler

def save_scaled_data(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Save scaled train and test data as CSV files
    
    Parameters:
    X_train, X_test: Original features
    X_train_scaled, X_test_scaled: Scaled features
    y_train, y_test: Target values
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save scaled training data
    scaled_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    scaled_train_df['CLASS'] = y_train.values
    scaled_train_df.to_csv("data/scaled_train.csv", index=False)
    
    # Save scaled testing data
    scaled_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    scaled_test_df['CLASS'] = y_test.values
    scaled_test_df.to_csv("data/scaled_test.csv", index=False)
    
    print("Scaled train and test data saved to 'data' directory")

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train Random Forest model
    
    Parameters:
    X_train: Training features
    y_train: Training target
    random_state (int): Random seed for reproducibility
    
    Returns:
    RandomForestClassifier: Trained model
    """
    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    
    print("Random Forest model trained and saved to 'models/random_forest_model.pkl'")
    
    return rf_model

def train_logistic_regression(X_train_scaled, y_train, random_state=42, max_iter=1000):
    """
    Train Logistic Regression model
    
    Parameters:
    X_train_scaled: Scaled training features
    y_train: Training target
    random_state (int): Random seed for reproducibility
    max_iter (int): Maximum iterations for solver
    
    Returns:
    LogisticRegression: Trained model
    """
    # Initialize and train the Logistic Regression model
    lr_model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    lr_model.fit(X_train_scaled, y_train)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    
    print("Logistic Regression model trained and saved to 'models/logistic_regression_model.pkl'")
    
    return lr_model

def run_training_pipeline(file_path="discretized.csv"):
    """
    Run the complete model training pipeline
    
    Parameters:
    file_path (str): Path to the CSV dataset
    
    Returns:
    tuple: Trained models and data splits
    """
    # Load data
    df = load_data(file_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    
    # Prepare scaled data for Logistic Regression
    X_train_scaled, X_test_scaled, scaler = prepare_scaled_data(X_train, X_test)
    
    # Train Logistic Regression model
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    
    return rf_model, lr_model, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Run the model training pipeline
    rf_model, lr_model, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = run_training_pipeline() 