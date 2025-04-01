import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for model evaluation
class ModelEvaluatingTech(ABC):
    @abstractmethod
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate a regression model."""
        pass

# Concrete evaluation class
class RegModelEvaluatingTech(ModelEvaluatingTech):
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate a trained model and return metrics."""
        logging.info("Making predictions.")
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mean_pred = np.mean(y_pred)  # Mean of predictions

        # Store metrics
        metrics = {
            "MSE": float(mse),
            "R²": float(r2),
            "Mean Prediction": float(mean_pred),
        }

        logging.info(f"Metrics: {metrics}")
        return metrics


# # Main execution
# if __name__ == "__main__":
#     # Generate sample data
#     np.random.seed(42)
#     X = pd.DataFrame(np.random.rand(100, 3), columns=["Feature1", "Feature2", "Feature3"])
#     y = pd.Series(3 * X["Feature1"] + 2 * X["Feature2"] + X["Feature3"] + np.random.randn(100))

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train a model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     evaluator = RegModelEvaluatingTech()
#     results = evaluator.evaluate_model(model, X_test, y_test)

#     # Print results
#     print(results)
