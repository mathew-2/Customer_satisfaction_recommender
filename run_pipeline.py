from zenml.client import Client

from pipeline.training_pipeline import train_pipeline


if __name__ == "__main__":

    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    data_path = "/Users/mathewmanoj/Documents/customer_satisfaction/data/olist_customers_dataset.csv"
    train_pipeline(data_path=data_path)