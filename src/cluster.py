import pandas as pd
from sklearn.preprocessing import StandardScaler


def data_load() -> pd.DataFrame:
    customer = pd.read_csv("data/Mall_Customers.csv")
    return customer

def preprocess(customer: pd.DataFrame):
    X = customer[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler