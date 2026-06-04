import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 


def data_load() -> pd.DataFrame:
    customer = pd.read_csv("data/Mall_Customers.csv")
    return customer

def preprocess(customer: pd.DataFrame)-> tuple:
    X = customer[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def find_optimal_k() -> int:
    return 5  # determined from elbow chart analysis
    
    