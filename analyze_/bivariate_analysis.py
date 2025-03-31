import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from abc import ABC, abstractmethod


class BivariateAnalysisTech(ABC):
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


class NumericalVsNumUnivariateAnalysisTech(BivariateAnalysisTech ):
    """
    Concrete class for analyzing numerical features.
    """

    def analysis(self, df: pd.DataFrame, feature1: str , feature2 : str):
        """
        Plots a histogram and KDE for a numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The numerical feature to analyze.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x = feature1 , y = feature2 , data = df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
    

        # Ensure directory exists before saving the plot
        
        save_path = f"/content/plots/{feature1} vs {feature2}_distribution.png"
        plt.savefig(save_path)

        print(f"Plot saved at: {save_path}")


class CategoricalVsNumUnivariateAnalysisTech(BivariateAnalysisTech):
    """
    Concrete class for analyzing categorical features.
    """

    def analysis(self, df: pd.DataFrame, feature1: str  , feature2 : str):
        """
        Plots a count plot for a categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The categorical feature to analyze.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x = feature1 , y = feature2 , data = df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
    

        # Ensure directory exists before saving the plot
        
        save_path = f"/content/plots/{feature1} vs {feature2}_distribution.png"
        plt.savefig(save_path)


        print(f"Plot saved at: {save_path}")


# def main():
#     """
#     Main function to handle command-line arguments and execute univariate analysis.
#     """
#     parser = argparse.ArgumentParser(description="Perform Bivariate analysis on a dataset.")
#     parser.add_argument("file_path", type=str, help="Path to the CSV dataset.")
#     parser.add_argument("feature1", type=str, help="Feature _ 1 column to analyze.")
#     parser.add_argument("feature2", type=str, help="Feature _ 2 column to analyze.")

    
  
#     parser.add_argument("--type", type=str, choices=["num_vs_num", "cat_vs_num"], required=True,
#                         help="Type of analysis: 'num_vs_num' (numerical vs numerical) or 'cat_vs_num' (categorical vs numerical).")

#     args = parser.parse_args()

#     # Load dataset
#     df = pd.read_csv(args.file_path)

#     # Select the correct analysis technique
#     if args.type ==  "num_vs_num":
#         analyzer = NumericalVsNumUnivariateAnalysisTech()
#     else:
#         analyzer = CategoricalVsNumUnivariateAnalysisTech()

#     # Perform analysis
#     analyzer.analysis(df, args.feature1, args.feature2)


# if __name__ == "__main__":
#     main()
