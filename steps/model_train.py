import logging
import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from steps.config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    config:ModelNameConfig = ModelNameConfig()) -> RegressorMixin:
    """we will train the model here on the ingested data
    Args:
        X_train : Training features
        X_test : Testing features
        Y_train : Training labels
        Y_test : Testing labels"""

    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train,Y_train)
            return trained_model
        else:
            raise ValueError(f"Model{config.model_name} not supported")
        
    except Exception as e:
        logging.error("Error in training model:{}".format(e))
        raise e
    
