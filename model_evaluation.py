import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib
import os

def load_models():
    """
    Load trained models from files
    
    Returns:
    tuple: Random Forest model, Logistic Regression model, scaler
    """
    rf_model = joblib.load('models/random_forest_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/robust_scaler.pkl')
    
    return rf_model, lr_model, scaler

def evaluate_random_forest(rf_model, X_test, y_test):
    """
    Evaluate Random Forest model
    
    Parameters:
    rf_model: Trained Random Forest model
    X_test: Test features
    y_test: Test target
    
    Returns:
    tuple: Predictions, accuracy, classification report, confusion matrix
    """
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)
    
    return y_pred, accuracy, report, cm

def evaluate_logistic_regression(lr_model, X_test_scaled, y_test):
    """
    Evaluate Logistic Regression model
    
    Parameters:
    lr_model: Trained Logistic Regression model
    X_test_scaled: Scaled test features
    y_test: Test target
    
    Returns:
    tuple: Predictions, accuracy, classification report, confusion matrix
    """
    # Make predictions
    y_pred = lr_model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)
    
    return y_pred, accuracy, report, cm

def plot_confusion_matrix(cm, model_name, class_names=None):
    """
    Plot confusion matrix
    
    Parameters:
    cm: Confusion matrix
    model_name: Name of the model (for title and filename)
    class_names: Names of the classes
    """
    # Create output directory if it doesn't exist
    os.makedirs("evaluation", exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    if class_names is None:
        class_names = ['Non-diabetic (0)', 'Diabetic (1)', 'Pre-diabetic (2)']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save the plot
    output_file = f"evaluation/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_file)
    print(f"Confusion matrix plot saved to {output_file}")
    
    plt.close()

def plot_feature_importance(rf_model, feature_names):
    """
    Plot feature importance for Random Forest model
    
    Parameters:
    rf_model: Trained Random Forest model
    feature_names: Names of the features
    """
    # Create output directory if it doesn't exist
    os.makedirs("evaluation", exist_ok=True)
    
    # Get feature importances
    feature_importance = rf_model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(feature_importance)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_feature_importance = feature_importance[indices]
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), sorted_feature_importance, align='center')
    plt.yticks(range(len(indices)), sorted_feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Random Forest Feature Importance')
    
    # Save the plot
    output_file = "evaluation/feature_importance.png"
    plt.savefig(output_file)
    print(f"Feature importance plot saved to {output_file}")
    
    plt.close()

def plot_roc_curve(model, X_test_scaled, y_test):
    """
    Plot ROC curve for multiclass classification
    
    Parameters:
    model: Trained model
    X_test_scaled: Scaled test features
    y_test: Test target
    """
    # Create output directory if it doesn't exist
    os.makedirs("evaluation", exist_ok=True)
    
    # Binarize the output labels for multiclass
    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Predict class probabilities
    y_pred_prob = model.predict_proba(X_test_scaled)
    
    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = {}, {}, {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    class_names = ['Non-diabetic', 'Diabetic', 'Pre-diabetic']
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, 
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right')
    
    # Save the plot
    output_file = "evaluation/roc_curve.png"
    plt.savefig(output_file)
    print(f"ROC curve plot saved to {output_file}")
    
    plt.close()
    
    return roc_auc

def save_evaluation_results(rf_results, lr_results):
    """
    Save evaluation results to CSV files
    
    Parameters:
    rf_results: Random Forest evaluation results (dict)
    lr_results: Logistic Regression evaluation results (dict)
    """
    # Create output directory if it doesn't exist
    os.makedirs("evaluation", exist_ok=True)
    
    # Combine results into DataFrames
    rf_df = pd.DataFrame(rf_results)
    lr_df = pd.DataFrame(lr_results)
    
    # Save to CSV
    rf_df.to_csv("evaluation/random_forest_results.csv", index=False)
    lr_df.to_csv("evaluation/logistic_regression_results.csv", index=False)
    
    print("Evaluation results saved to 'evaluation' directory")

def run_evaluation_pipeline(X_test=None, X_test_scaled=None, y_test=None):
    """
    Run the complete model evaluation pipeline
    
    Parameters:
    X_test: Test features (optional)
    X_test_scaled: Scaled test features (optional)
    y_test: Test target (optional)
    
    Returns:
    tuple: Evaluation results for both models
    """
    # Load models
    rf_model, lr_model, scaler = load_models()
    
    # If test data is not provided, load from CSV
    if X_test is None or y_test is None:
        test_data = pd.read_csv("data/test_data.csv")
        X_test = test_data.drop('CLASS', axis=1)
        y_test = test_data['CLASS']
    
    # If scaled test data is not provided, scale using loaded scaler
    if X_test_scaled is None:
        X_test_scaled = scaler.transform(X_test)
    
    # Evaluate Random Forest
    rf_y_pred, rf_accuracy, rf_report, rf_cm = evaluate_random_forest(rf_model, X_test, y_test)
    
    # Evaluate Logistic Regression
    lr_y_pred, lr_accuracy, lr_report, lr_cm = evaluate_logistic_regression(lr_model, X_test_scaled, y_test)
    
    # Plot confusion matrices
    plot_confusion_matrix(rf_cm, "Random Forest")
    plot_confusion_matrix(lr_cm, "Logistic Regression")
    
    # Plot feature importance for Random Forest
    plot_feature_importance(rf_model, X_test.columns)
    
    # Plot ROC curve for Logistic Regression
    roc_auc = plot_roc_curve(lr_model, X_test_scaled, y_test)
    
    # Prepare results
    rf_results = {
        'model': ['Random Forest'] * len(rf_report),
        'class': list(rf_report.keys()),
        'precision': [rf_report[c]['precision'] if c in rf_report else None for c in rf_report],
        'recall': [rf_report[c]['recall'] if c in rf_report else None for c in rf_report],
        'f1_score': [rf_report[c]['f1-score'] if c in rf_report else None for c in rf_report],
        'accuracy': [rf_accuracy] * len(rf_report)
    }
    
    lr_results = {
        'model': ['Logistic Regression'] * len(lr_report),
        'class': list(lr_report.keys()),
        'precision': [lr_report[c]['precision'] if c in lr_report else None for c in lr_report],
        'recall': [lr_report[c]['recall'] if c in lr_report else None for c in lr_report],
        'f1_score': [lr_report[c]['f1-score'] if c in lr_report else None for c in lr_report],
        'accuracy': [lr_accuracy] * len(lr_report),
        'roc_auc': [roc_auc.get(i, None) for i in range(3)] + [None] * (len(lr_report) - 3)
    }
    
    # Save evaluation results
    save_evaluation_results(rf_results, lr_results)
    
    return rf_results, lr_results

if __name__ == "__main__":
    # Run the evaluation pipeline
    rf_results, lr_results = run_evaluation_pipeline() 