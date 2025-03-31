import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from abc import ABC, abstractmethod

# Abstract base class for missing values analysis
class MissingValuesAnalysisTemp(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame):
        """
        Identify and visualize missing values in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The data to be analyzed.

        Returns:
        None (Prints missing values and displays a heatmap).
        """
        pass

# Concrete implementation of missing values analysis
class MissingValuesAnalysis(MissingValuesAnalysisTemp):
    def analyze(self, df: pd.DataFrame):
        """
        Perform both identification and visualization of missing values.

        Parameters:
        df (pd.DataFrame): The data to be analyzed.

        Returns:
        None (Prints missing values and displays a heatmap).
        """
        # Identify missing values
        missing_values = df.isnull().sum()
        print("### Missing Values Analysis ###")
        print(missing_values)

        # Convert Boolean missing values to numerical (1 = Missing, 0 = Not Missing)
        missing_matrix = df.isnull().astype(int)

        # Visualize missing values using a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(missing_matrix, cbar=False, cmap="viridis", yticklabels=False)
        plt.title("Missing Values Heatmap")
        save_path = "/content/plots/missing_values_heatmap.png" 
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
       # plt.show() 
        

# Main script execution with argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze missing values in a CSV file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file.")
    args = parser.parse_args()

    # Load the CSV file
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        exit(1)

    # Perform missing values analysis
    analyzer = MissingValuesAnalysis()
    analyzer.analyze(df)
