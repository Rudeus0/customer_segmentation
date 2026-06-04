import pandas as pd


def data_load() -> pd.DataFrame:
    customer = pd.read_csv("data/Mall_Customers.csv")
    return customer