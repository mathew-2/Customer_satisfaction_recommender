import logging
import pandas as pd
import numpy as np

from typing import Union
from abc import ABC , abstractmethod

from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for data cleaning
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass



class DataPreprocessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        preprocessing data 
        """
        try:

            cols_to_drop = [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_purchase_timestamp"
            ]
            data = data.drop(columns=[
                c for c in cols_to_drop if c in data.columns],
                             axis = 1)

            # data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            # data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            # data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            # data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            dimension_cols = [
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]

            for col in dimension_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())

            # data["review_comment_message"].fillna("No comment", inplace=True)

            if "review_comment_message" in data.columns:
                data["review_comment_message"] = data[
                    "review_comment_message"
                ].fillna("No comment")
            if "review_score" in data.columns:
                data["review_score"] = pd.to_numeric(data["review_score"], errors="coerce")
            data = data.select_dtypes(include = [np.number])
            # cols_to_drop = ["customer_zip_code_prefix","order_item_id"]
            # data = data.drop(cols_to_drop,axis = 1)

            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]

            data = data.drop(
                columns=[c for c in cols_to_drop if c in data.columns]
            )
            return data
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e
        
class DataDivideStrategy(DataStrategy):
    """Dividing dataset into train and test set"""
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            x = data.drop(["review_score"],axis = 1)
            y = data["review_score"]

            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
            return x_train,x_test,y_train,y_test
        except Exception as e:
            logging.error(f"Error in data dividing: {e}")
            raise e
        

class DataCleaning:
    """ class for cleaning the data and processes into train and test set""" 
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        "handle data"
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
