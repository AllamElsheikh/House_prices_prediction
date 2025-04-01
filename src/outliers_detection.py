import logging
from abc import ABC , abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)


class OutlierDetcaitonTech(ABC):
    @abstractmethod
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class ZScoreOutlierDetection(OutlierDetcaitonTech):
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using Z-Score method.")
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = (z_scores > self.threshold).any(axis=1)
        return outliers


class IQROutlierDetection(OutlierDetcaitonTech):
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method.")
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
        return outliers


class OutlierDetector:

  def __init__(self, strategy : OutlierDetcaitonTech):
    self.strategy = strategy

  def set_strategy(self  , strategy : OutlierDetcaitonTech):
    self.strategy = strategy
  def detect_outliers (self , df :pd.DataFrame):
    return self.strategy.detect_outliers(df)

  def handling_outliers(self , df :pd.DataFrame , method : str = "remove" , **kwarg) ->pd.DataFrame :
    outliers =   self.detect_outliers(df)
    if method == "remove":
      df_cln = df[~outliers]
      return df_cln

    elif method == "cap" :
      df_cln = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
      return df_cln
    else :
      logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
      return df

  def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")


# # Example usage
# if __name__ == "__main__":
#     # Example dataframe
#     df = pd.read_csv("/content/House_prices_prediction/extracted_data/AmesHousing.csv")
#     df_numeric = df.select_dtypes(include=[np.number]).dropna()

#     # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
#     outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

#     # Detect and handle outliers
#     outliers = outlier_detector.detect_outliers(df_numeric)
#     df_cleaned = outlier_detector.handling_outliers(df_numeric, method="remove")

#     print(df_cleaned.shape)
#     # Visualize outliers in specific features
#     # outlier_detector.visualize_outliers(df_cleaned, features=["SalePrice", "Gr Liv Area"])
#     pass

