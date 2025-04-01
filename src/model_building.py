import logging
from abc import ABC , abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BuildTrainModel(ABC):
    @abstractmethod
    def build_tarin_model(self , X_train  : pd.DataFrame , y_train : pd.Series) -> RegressorMixin:
        pass


class LinearRegressionModel(BuildTrainModel):
   def build_tarin_model(self , X_train  : pd.DataFrame , y_train : pd.Series) -> Pipeline:
     if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
     if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

     logging.info("Initializing Linear Regression model with scaling.")

     pipline = Pipeline(
         [
             ("scaler", StandardScaler()),
             ("regressor", LinearRegression()),
         ]
     )

     logging.info("Model training completed.")
     return pipline

     