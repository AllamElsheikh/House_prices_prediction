import argparse
import pandas as pd
from abc import ABC, abstractmethod

# Abstract base class for data inspection techniques
class DataInspectionTech(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Execute a specific inspection on the data.

        Parameters:
        df (pd.DataFrame): The data to be inspected.

        Returns:
        None (Prints the inspected data).
        """
        pass

# Strategy for displaying data types and non-null counts
class DataTypeTech(DataInspectionTech):
    def inspect(self, df: pd.DataFrame):
        """
        Print the data type and non-null count for each column.

        Parameters:
        df (pd.DataFrame): The data to be inspected.

        Returns:
        None (Prints the inspected data).
        """
        print("\n### Data Types and Non-null Counts ###")
        print(df.info())

# Strategy for displaying mathematical and statistical descriptions
class MathDescribtionTech(DataInspectionTech):
    def inspect(self, df: pd.DataFrame):
        """
        Print the mathematical and statistical description for each column.

        Parameters:
        df (pd.DataFrame): The data to be inspected.

        Returns:
        None (Prints summary statistics).
        """
        print("\n### Summary Statistics (Numerical Features) ###")
        print(df.describe())
        print("\n### Summary Statistics (Categorical Features) ###")
        print(df.describe(include=["O"]))

# Context class that applies different inspection strategies
class DataInspector:
    def __init__(self, Tech: DataInspectionTech):
        """
        Initialize the DataInspector with a specific technique.

        Parameters:
        Tech (DataInspectionTech): The initial strategy for inspecting the data.

        Returns:
        None
        """
        self.Tech = Tech

    def set_strategy(self, Tech: DataInspectionTech):
        """
        Set a new strategy for data inspection.

        Parameters:
        Tech (DataInspectionTech): The new strategy to be used.

        Returns:
        None
        """
        self.Tech = Tech

    def execute_inspection(self, df: pd.DataFrame):
        """
        Execute the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The DataFrame to be inspected.

        Returns:
        None (Executes the strategy's inspection method).
        """
        self.Tech.inspect(df)

# Main script execution with argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform data inspection on a given CSV file.")
    
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file.")
    parser.add_argument("--tech", type=str, choices=["type", "stats"], required=True, help="Inspection type: 'type' for data types, 'stats' for summary statistics.")
    
    args = parser.parse_args()

    # Load the CSV file
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        exit(1)

    # Select the appropriate strategy
    if args.tech == "type":
        inspector = DataInspector(DataTypeTech())
    elif args.tech == "stats":
        inspector = DataInspector(MathDescribtionTech())

    # Execute inspection
    inspector.execute_inspection(df)
