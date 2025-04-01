import pandas as pd
from zenml import step
from typing import Tuple
import logging
from sklearn.pipeline import Pipeline
import sys 
sys.path.append("/content/House_prices_prediction/src")
from model_evaluate import RegModelEvaluatingTech


@step(enable_cache=False)
def model_evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple [dict , float]:

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying the same preprocessing to the test data.")

    x_test_pre = model.named_step['preprocessor'].transform(X_test)

    evaluation_metrics = RegModelEvaluatingTech.evaluate_model(model.named_steps["model"], x_test_pre, y_test)

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse = evaluation_metrics.get("Mean Squared Error", None)
    return evaluation_metrics, mse
