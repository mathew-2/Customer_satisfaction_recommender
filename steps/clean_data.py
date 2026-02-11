import logging
import pandas as pd
from zenml import step

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """clean the data by handling missing values and duplicates"""
    pass