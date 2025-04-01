import logging  # Import logging for tracking events
import pandas as pd  # Import pandas for data handling
import numpy as np  # Import NumPy for numerical operations
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting datasets
import argparse  # Import argparse to handle command-line arguments
from abc import ABC, abstractmethod  # Import ABC for defining abstract classes

# Configure logging settings to display messages with timestamps and log levels
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract base class for different data splitting techniques
class DataSplittingTech(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, target_col: str):
        """
        Abstract method for splitting the dataset.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            target_col (str): The column name of the target variable.

        Returns:
            Tuple: Splitted train and test datasets.
        """
        pass


# Concrete implementation of a simple train-test split
class SimpleSplitting(DataSplittingTech):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize SimpleSplitting with test size and random state.

        Args:
            test_size (float): Proportion of the dataset to be used as test data.
            random_state (int): Random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_col: str):
        """
        Splits the data into training and testing sets.

        Args:
            df (pd.DataFrame): The input dataset.
            target_col (str): The column name of the target variable.

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logging.info("Performing simple train-test split.")

        # Separate features (X) and target variable (y)
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Perform the train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test


# Wrapper class that allows using different splitting techniques
class DataSplitter:
    def __init__(self, splitting_tech: DataSplittingTech):
        """
        Initialize the DataSplitter with a specific splitting technique.

        Args:
            splitting_tech (DataSplittingTech): An instance of a data splitting strategy.
        """
        self.splitting_tech = splitting_tech

    def split(self, df: pd.DataFrame, target_col: str):
        """
        Splits the data using the specified splitting technique.

        Args:
            df (pd.DataFrame): The input dataset.
            target_col (str): The column name of the target variable.

        Returns:
            Tuple: Splitted train and test datasets.
        """
        return self.splitting_tech.split(df, target_col)


# Execute the script only when run directly
if __name__ == "__main__":
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Perform train-test split on a dataset.")
    parser.add_argument("file_path", type=str, help="Path to the dataset file (CSV format).")
    parser.add_argument("target_column", type=str, help="The target column name for prediction.")
    
    args = parser.parse_args()

    # Read dataset from the provided file path
    try:
        df = pd.read_csv(args.file_path)
        logging.info(f"Dataset loaded successfully from {args.file_path}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        exit(1)

    # Initialize the splitting technique
    splitter = DataSplitter(SimpleSplitting(test_size=0.2, random_state=42))

    # Perform the split
    X_train, X_test, y_train, y_test = splitter.split(df, args.target_column)

    # Log dataset shapes after splitting
    logging.info(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Save the train-test datasets
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    logging.info("Train-test datasets saved as CSV files.")
