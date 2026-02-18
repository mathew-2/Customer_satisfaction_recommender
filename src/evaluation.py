import logging
from abc import ABC ,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error

class Evaluation(ABC):
    """ Abstract class defining strategies to evaluate our model"""

    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """calculates the multiple scores of a model 
        Args:
            y_true: True Labels
            y_pred: predicted Labels
        Returns 
            None
        """

        pass


class MSE(Evaluation):
    """Evaluation strategy that calculates the Mean Square Error"""

    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray):

        try:
            logging.info("calculating the MSE Loss")
            mse = mean_squared_error(y_true,y_pred)
            logging.info(f"The MSE loss is {mse}")
            return mse

        except Exception as e:
            logging.error(f"Error in calculating the MSE loss{e}")
            raise e
        
class R2(Evaluation):
    """Evaluation strategy that calculates the r2 score"""

    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray):

        try:
            logging.info("calculating the r2 score")
            r2 = r2_score(y_true,y_pred)
            logging.info(f"The R2 score is {r2}")
            return r2

        except Exception as e:
            logging.error(f"Error in calculating the r2 score : {e}")
            raise e
        

class RMSE(Evaluation):
    """Evaluation strategy that calculates the rmse loss value"""

    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray):

        try:
            logging.info("calculating the RMSE value")
            rmse = root_mean_squared_error(y_true,y_pred)
            logging.info(f"The RMSE is {rmse}")
            return rmse

        except Exception as e:
            logging.error(f"Error in calculating the r2 score : {e}")
            raise e
