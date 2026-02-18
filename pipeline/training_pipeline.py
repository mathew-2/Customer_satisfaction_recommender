from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    """Pipeline for training the model"""
    df = ingest_data(data_path)
    # clean_data(df)
    X_train,X_test,Y_train,Y_test = clean_data(df)
    # train_model(df)
    trained_model = train_model(X_train,X_test,Y_train,Y_test)
    # evaluate_model(df)
    r2_score,rmse = evaluate_model(trained_model,X_test,Y_test)



