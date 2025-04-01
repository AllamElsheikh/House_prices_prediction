import pandas as pd
import sys
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
import mlflow
from typing import Annotated
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Setup MLflow experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    logging.info(f"Categorical columns: {categorical_features.tolist()}")
    logging.info(f"Numerical columns: {numerical_features.tolist()}")

    # Define preprocessing steps
    num_transformer = SimpleImputer(strategy="mean")
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # Ensure an active MLflow run before logging
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable MLflow autologging
        mlflow.sklearn.autolog()

        logging.info("Building and training the Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Log expected columns after encoding
        onehot_encoder = pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        onehot_encoder.fit(X_train[categorical_features])

        expected_columns = numerical_features.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_features)
        )
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline
