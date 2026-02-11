import logging

import pandas as pd
from zenml import step

@step
def evaluate_model(data: pd.DataFrame) -> None:
    """Evaluate the trained model here"""
    pass