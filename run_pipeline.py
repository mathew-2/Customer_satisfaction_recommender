from pipeline.training_pipeline import train_pipeline


if __name__ == "__main__":
    data_path = "/Users/mathewmanoj/Documents/customer_satisfaction/data/olist_customers_dataset.csv"
    train_pipeline(data_path=data_path)