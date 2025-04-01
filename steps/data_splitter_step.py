import pandas as pd  # Importing pandas for data manipulation
import sys  # Importing sys to modify system path
from typing import Tuple  # Importing Tuple for type hinting

# Adding the path to the directory containing the 'data_splitter' module
sys.path.append('/content/House_prices_prediction/src') 

# Importing necessary classes for data splitting from 'data_splitter' module
from data_splitter import SimpleSplitting, DataSplitter

# Importing the 'step' decorator from ZenML for pipeline step definition
from zenml import step


@step 
def data_splitter_step(
    df: pd.DataFrame,  # Input parameter: 'df' is the dataset to be split, as a pandas DataFrame
    target_col: str    # Input parameter: 'target_col' is the name of the target column (the variable to predict)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:  # Output: A tuple containing training and test splits
    """
    This function splits the input dataset into training and test sets, based on the provided target column.
    It uses the SimpleSplitting strategy to split the data.

    Args:
        df (pd.DataFrame): The input dataset that will be split.
        target_col (str): The column name representing the target variable to predict.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        A tuple containing the training and test sets for features (X) and target (y):
            - X_train: Features for training
            - X_test: Features for testing
            - y_train: Target for training
            - y_test: Target for testing
    """
    
    # Creating an instance of the DataSplitter with the SimpleSplitting strategy
    splitter = DataSplitter(SimpleSplitting())
    
    # Using the splitter to split the data into training and test sets
    X_train, X_test, y_train, y_test = splitter.split(df, target_col) 
    
    # Returning the resulting train-test splits for features and target
    return X_train, X_test, y_train, y_test
