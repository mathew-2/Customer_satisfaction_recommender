import logging

import pandas as pd
from zenml import step


class IngestData:
    """ingesting data from the data path"""

    def __init__(self,data_path:str):
        self.data_path = data_path


    def get_data(self):
        logging.info(f"ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """Ingesting data from the data path"""

    try:
        ingest_data = IngestData(data_path)
        data = ingest_data.get_data()
        logging.info("Data ingestion completed successfully.")
        return data
    
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise e