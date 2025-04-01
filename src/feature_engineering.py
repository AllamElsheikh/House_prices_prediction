import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import argparse  # Importing argparse for command-line argument parsing

# Setting up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineerginTech(ABC):
    """Abstract class for feature engineering techniques"""
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to apply transformation to the dataset"""
        pass


class LogTransformation(FeatureEngineerginTech):
    """Log transformation feature engineering class"""
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies log transformation (log1p) to specified features"""
        logging.info(f"Applying log transformation to features {self.features}")
        df_trains = df.copy()
        for feature in self.features:
            df_trains[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_trains


class MinMaxScalerTransformation(FeatureEngineerginTech):
    """MinMaxScaler transformation class"""
    def __init__(self, features, feature_range=(0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies MinMaxScaler transformation to specified features"""
        logging.info(f"Applying MinMaxScaler transformation to features {self.features}")
        df_trains = df.copy()
        df_trains[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("MinMaxScaler transformation completed.")
        return df_trains


class StandardScalerTransformation(FeatureEngineerginTech):
    """StandardScaler transformation class"""
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies StandardScaler transformation to specified features"""
        logging.info(f"Applying StandardScaler transformation to features {self.features}")
        df_trains = df.copy()
        df_trains[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("StandardScaler transformation completed.")
        return df_trains


class OneHotEncoderTransformation(FeatureEngineerginTech):
    """OneHotEncoder transformation class"""
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")  # Drop first to avoid multicollinearity

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies OneHotEncoder transformation to specified categorical features"""
        logging.info(f"Applying OneHotEncoder transformation to features {self.features}")
        df_trains = df.copy()

        # Encode features and create a DataFrame with the encoded values
        encoded_df = pd.DataFrame(self.encoder.fit_transform(df[self.features]), 
                                  columns=self.encoder.get_feature_names_out(self.features))
        # Drop the original categorical features and concatenate the new one-hot encoded features
        df_trains = df_trains.drop(columns=self.features).reset_index(drop=True)
        df_trains = pd.concat([df_trains, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_trains

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineerginTech):
       
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineerginTech):
        
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)



# # Main entry point of the script
# if __name__ == "__main__":
#     # Create the argument parser
#     parser = argparse.ArgumentParser(description="Apply feature engineering transformations")
#     parser.add_argument('file_path', type=str, help="Path to the input CSV file")
#     parser.add_argument('transformation_type', type=str, choices=['log', 'minmax', 'standard', 'onehot'], 
#                         help="Type of transformation to apply")
#     parser.add_argument('features', type=str, nargs='+', help="List of features to apply transformation on")

#     # Parse the command-line arguments
#     args = parser.parse_args()

#     # Load the data from the specified file path
#     df = pd.read_csv(args.file_path)

#     # Apply the chosen transformation
#     if args.transformation_type == 'log':
#         transformer = LogTransformation(features=args.features)
#     elif args.transformation_type == 'minmax':
#         transformer = MinMaxScalerTransformation(features=args.features)
#     elif args.transformation_type == 'standard':
#         transformer = StandardScalerTransformation(features=args.features)
#     elif args.transformation_type == 'onehot':
#         transformer = OneHotEncoderTransformation(features=args.features)

#     # Apply the transformation
#     transformed_df = transformer.apply_transformation(df)

#     # Save the transformed data
#     transformed_df.to_csv("transformed_data.csv", index=False)
#     logging.info("Transformation complete and saved to transformed_data.csv")
