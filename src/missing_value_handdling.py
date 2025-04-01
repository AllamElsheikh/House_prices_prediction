import logging
from abc import ABC, abstractmethod
import pandas as pd
import argparse

# Setting up logging to capture and display logs with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for handling missing values
class MissValueHandlingTech(ABC):
    @abstractmethod
    def handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method for handling missing values."""
        pass

# Class to drop rows or columns with missing values
class DropMissingValues(MissValueHandlingTech):
    def __init__(self, axis=0, thresh=None):
        """
        Initialize with axis (0 for rows, 1 for columns) and thresh (minimum non-NA values).
        """
        self.axis = axis
        self.thresh = thresh

    def handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop missing values from the DataFrame based on the axis and threshold.
        """
        logging.info("Dropping missing values...")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Dropping missing values completed.")
        return df_cleaned

# Class to fill missing values with specified methods (mean, median, mode, constant)
class FillMissingValues(MissValueHandlingTech):
    def __init__(self, method: str = "mean", fill_value=None):
        """
        Initialize with the method (mean, median, mode, constant) and the fill_value for constant.
        """
        self.method = method
        self.fill_value = fill_value

    def handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame using the specified method.
        """
        logging.info("Filling missing values...")
        df_cleaned = df.copy()

        # Handling missing values using different strategies
        if self.method == "mean":
            num_cols = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].mean())
        elif self.method == "median":
            num_cols = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())
        elif self.method == "mode":
            num_cols = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].mode().iloc[0])
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")
        
        logging.info("Missing values filled.")
        return df_cleaned

# Main function to run the script
if __name__ == "__main__":
    # Argument parser to take command-line arguments
    parser = argparse.ArgumentParser(description="Handle missing values in a DataFrame.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the cleaned CSV file.")
    parser.add_argument("--method", type=str, choices=["mean", "median", "mode", "constant"], default="mean", help="Method for filling missing values.")
    parser.add_argument("--axis", type=int, choices=[0, 1], default=0, help="Axis for dropping missing values (0 for rows, 1 for columns).")
    parser.add_argument("--thresh", type=int, default=None, help="Minimum non-NA values required to retain a row or column.")

    # Parsing arguments
    args = parser.parse_args()

    # Load the data from the provided input file
    df = pd.read_csv(args.input_file)
    
    # Decide whether to drop or fill missing values
    if args.method in ["mean", "median", "mode", "constant"]:
        handler = FillMissingValues(method=args.method, fill_value=None if args.method != "constant" else 0)
    else:
        handler = DropMissingValues(axis=args.axis, thresh=args.thresh)
    
    # Apply the transformation
    df_cleaned = handler.handling(df)

    # Save the cleaned data to the output file
    df_cleaned.to_csv(args.output_file, index=False)
    logging.info(f"Transformation complete. Saved to {args.output_file}")
