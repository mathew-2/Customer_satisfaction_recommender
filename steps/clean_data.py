import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataPreprocessingStrategy,DataDivideStrategy,DataCleaning

@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"Y_train"],
    Annotated[pd.Series,"Y_test"],
]:
    """clean the data by handling missing values and duplicates
    Args:
        data : Raw data to be cleaned
    
    Returns:
        x_train : Training features
        x_test : Testing features
        y_train : Training labels
        y_test : Testing labels
    
    """
    try:
        preprocess_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(data,preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        dividing_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,dividing_strategy)
        x_train,x_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Data cleaned successfully")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e


