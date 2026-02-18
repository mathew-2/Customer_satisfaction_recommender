import logging

import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from typing import Annotated
from typing_extensions import Tuple

from src.evaluation import MSE,RMSE,R2


@step
def evaluate_model(model:RegressorMixin,
                   X_test:pd.DataFrame,
                   Y_test:pd.DataFrame) -> Tuple[
                       Annotated[float,"r2_score"],
                       Annotated[float,"rmse"]
                   ]:
    """Evaluate the trained model here on the ingested data
    
    Args:
        X_true: True dataset X
        Y_true: Ground Truth Labels
    Returns:
        R2_Score(float) and RMSE Loss """
    
    try:
        prediction = model.predict(X_test)
        MSE_class = MSE()
        MSE_loss = MSE_class.calculate_scores(Y_test,prediction)

        RMSE_class= RMSE()
        rmse_loss = RMSE_class.calculate_scores(Y_test,prediction)

        R2_class = R2()
        r2_score = R2_class.calculate_scores(Y_test,prediction)

        logging.info("scores calculated successfully")

        return r2_score,rmse_loss
    
    except Exception as e:
        logging.error(f"error in evalution of model {e}")
        raise e
    