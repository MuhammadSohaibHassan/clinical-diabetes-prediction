import os
import argparse
from data_preprocessing import run_eda_pipeline
from model_training import run_training_pipeline
from model_evaluation import run_evaluation_pipeline
import subprocess

def main(args):
    """
    Run the complete pipeline: preprocessing, training, evaluation, and optionally the app
    
    Parameters:
    args: Command-line arguments
    """
    print("=" * 80)
    print("Diabetes Prediction System Pipeline")
    print("=" * 80)
    
    # Step 1: Data Preprocessing
    if args.preprocess:
        print("\n[1/3] Running Data Preprocessing and EDA...")
        df = run_eda_pipeline(args.data_path)
        print("Data preprocessing complete!")
    
    # Step 2: Model Training
    if args.train:
        print("\n[2/3] Training Models...")
        models = run_training_pipeline(args.data_path)
        print("Model training complete!")
    
    # Step 3: Model Evaluation
    if args.evaluate:
        print("\n[3/3] Evaluating Models...")
        results = run_evaluation_pipeline()
        print("Model evaluation complete!")
    
    # Step 4: Launch Streamlit app (optional)
    if args.app:
        print("\nLaunching Streamlit Application...")
        try:
            subprocess.run(["streamlit", "run", "app.py"])
        except Exception as e:
            print(f"Error launching Streamlit app: {e}")
            print("Try running manually with: streamlit run app.py")
    
    print("\nPipeline execution completed successfully!")
    
    if not any([args.preprocess, args.train, args.evaluate, args.app]):
        print("\nNo actions specified. Use arguments to run specific parts of the pipeline:")
        print("  --preprocess: Run data preprocessing and EDA")
        print("  --train: Train the models")
        print("  --evaluate: Evaluate trained models")
        print("  --app: Launch the Streamlit application")
        print("  --all: Run the complete pipeline")
        print("\nExample: python run.py --all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diabetes Prediction System Pipeline")
    
    parser.add_argument("--data_path", type=str, default="discretized.csv",
                        help="Path to the dataset CSV file")
    
    parser.add_argument("--preprocess", action="store_true", 
                        help="Run data preprocessing and EDA")
    
    parser.add_argument("--train", action="store_true",
                        help="Train the models")
    
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate trained models")
    
    parser.add_argument("--app", action="store_true",
                        help="Launch the Streamlit application")
    
    parser.add_argument("--all", action="store_true",
                        help="Run the complete pipeline")
    
    args = parser.parse_args()
    
    # If --all is specified, set all flags to True
    if args.all:
        args.preprocess = True
        args.train = True
        args.evaluate = True
        args.app = True
    
    main(args) 