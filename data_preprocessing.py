import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_data(file_path="discretized.csv"):
    """Load the dataset from CSV"""
    return pd.read_csv(file_path)

def create_output_directory(output_folder="eda_output"):
    """Create directory for saving EDA outputs"""
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def plot_distributions(df, output_folder="eda_output"):
    """
    Plot histograms for each numerical feature
    
    Parameters:
    df (DataFrame): The input dataframe
    output_folder (str): Folder to save plots
    """
    create_output_directory(output_folder)
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, color="skyblue", bins=30)
        plt.title(f"Distribution of {col}")
        output_file = os.path.join(output_folder, f"distribution_{col}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Distribution plot for {col} saved to {output_file}")

def plot_correlation_matrix(df, output_folder="eda_output"):
    """
    Create and save correlation matrix heatmap
    
    Parameters:
    df (DataFrame): The input dataframe
    output_folder (str): Folder to save plots
    """
    create_output_directory(output_folder)
    
    # Correlation matrix
    correlation_matrix = df.corr()

    # Save correlation matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    output_file = os.path.join(output_folder, "correlation_matrix.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Correlation matrix saved to {output_file}")

def detect_outliers(df):
    """
    Detect outliers using the IQR method
    
    Parameters:
    df (DataFrame): The input dataframe
    
    Returns:
    dict: Dictionary containing outlier counts for each numerical column
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    outlier_counts = {}
    
    print("Outlier counts using IQR method:")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)  # First quartile
        Q3 = df[col].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile Range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = outliers_count
        print(f"{col}: {outliers_count} outliers")
    
    return outlier_counts

def plot_boxplots(df, output_folder="eda_output"):
    """
    Create boxplots for each numerical feature to visualize outliers
    
    Parameters:
    df (DataFrame): The input dataframe
    output_folder (str): Folder to save plots
    """
    create_output_directory(output_folder)
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        output_file = os.path.join(output_folder, f"boxplot_{col}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Boxplot for {col} saved to {output_file}")

def run_eda_pipeline(file_path="discretized.csv", output_folder="eda_output"):
    """
    Run the complete EDA pipeline
    
    Parameters:
    file_path (str): Path to the CSV dataset
    output_folder (str): Folder to save plots and outputs
    """
    # Load data
    df = load_data(file_path)
    
    # Create output directory
    create_output_directory(output_folder)
    
    # Generate plots
    plot_distributions(df, output_folder)
    plot_correlation_matrix(df, output_folder)
    plot_boxplots(df, output_folder)
    
    # Detect outliers
    outlier_counts = detect_outliers(df)
    
    # Print dataset info
    print("\nDataset Info:")
    print(df.info())
    
    # Print descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    # Run the EDA pipeline
    df = run_eda_pipeline() 