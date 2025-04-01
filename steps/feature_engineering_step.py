import pandas as pd
import sys
import argparse  # Import argparse for command-line argument parsing
from zenml import step
# Add the feature engineering module path to sys.path
sys.path.append('/content/House_prices_prediction/src')

# Importing feature engineering transformations
from feature_engineering import ( 
    OneHotEncoderTransformation,
    StandardScalerTransformation,
    MinMaxScalerTransformation,
    LogTransformation
)

# Define the ZenML step
@step 
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    """
    Perform feature engineering transformations on the input DataFrame based on the selected strategy.
    
    Parameters:
    - df: The input DataFrame
    - strategy: The feature engineering strategy to apply (log, minmax, standard, onehot)
    - features: List of features to apply the transformation on
    
    Returns:
    - Transformed DataFrame
    """
    # Validate input features
    if features is None:
        raise ValueError("Features cannot be None")
    
    # Apply transformation based on the chosen strategy
    if strategy == "log":
        return LogTransformation(features).apply_transformation(df)
    elif strategy == "minmax":
        return MinMaxScalerTransformation(features).apply_transformation(df)
    elif strategy == "standard":
        return StandardScalerTransformation(features).apply_transformation(df)
    elif strategy == "onehot":
        return OneHotEncoderTransformation(features).apply_transformation(df)
    else:
        raise ValueError(f"Unknown strategy {strategy}. Choose from 'log', 'minmax', 'standard', or 'onehot'.")

# Main entry point of the script
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Apply feature engineering transformations to a dataset")
    
    # Arguments for the script
    parser.add_argument('file_path', type=str, help="Path to the input CSV file")
    parser.add_argument('strategy', type=str, choices=['log', 'minmax', 'standard', 'onehot'], 
                        help="Transformation strategy to apply (log, minmax, standard, onehot)")
    parser.add_argument('features', type=str, nargs='+', help="List of features to apply transformation on")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Load the data from the specified file path
    df = pd.read_csv(args.file_path)
    
    # Apply feature engineering transformation based on strategy
    transformed_df = feature_engineering_step(df, strategy=args.strategy, features=args.features)
    
    # Save the transformed data to a new file
    transformed_df.to_csv("transformed_data.csv", index=False)
    print("Transformation complete. Saved to transformed_data.csv")
