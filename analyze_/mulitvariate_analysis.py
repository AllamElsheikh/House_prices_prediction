import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from abc import ABC, abstractmethod


class UnivariateAnalysisTech(ABC):
    """
    Abstract base class for univariate analysis techniques.
    """

    @abstractmethod
    def analysis(self, df: pd.DataFrame, feature: str):
        """
        Abstract method to analyze a given feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The feature column to analyze.
        """
        pass


class NumericalUnivariateAnalysisTech(UnivariateAnalysisTech):
    """
    Concrete class for analyzing numerical features.
    """

    def analysis(self, df: pd.DataFrame, feature: str):
        """
        Plots a histogram and KDE for a numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The numerical feature to analyze.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=6)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        # Ensure directory exists before saving the plot
        
        save_path = f"/content/plots/{feature}_distribution.png"
        plt.savefig(save_path)
        plt.show()

        print(f"Plot saved at: {save_path}")


class CategoricalUnivariateAnalysisTech(UnivariateAnalysisTech):
    """
    Concrete class for analyzing categorical features.
    """

    def analysis(self, df: pd.DataFrame, feature: str):
        """
        Plots a count plot for a categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The categorical feature to analyze.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[feature])  # Removed kde=True and bins=6 (not applicable)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")

        # Ensure directory exists before saving the plot
      
        save_path =f"/content/plots/cat_{feature}_distribution.png"
        plt.savefig(save_path)
        plt.show()

        print(f"Plot saved at: {save_path}")


# def main():
#     """
#     Main function to handle command-line arguments and execute univariate analysis.
#     """
#     parser = argparse.ArgumentParser(description="Perform univariate analysis on a dataset.")
#     parser.add_argument("file_path", type=str, help="Path to the CSV dataset.")
#     parser.add_argument("feature", type=str, help="Feature column to analyze.")
#     parser.add_argument("--type", type=str, choices=["numerical", "categorical"], required=True,
#                         help="Type of feature: 'numerical' or 'categorical'.")

#     args = parser.parse_args()

#     # Load dataset
#     df = pd.read_csv(args.file_path)

#     # Select the correct analysis technique
#     if args.type == "numerical":
#         analyzer = NumericalUnivariateAnalysisTech()
#     else:
#         analyzer = CategoricalUnivariateAnalysisTech()

#     # Perform analysis
#     analyzer.analysis(df, args.feature)


# if __name__ == "__main__":
#     main()
