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

def fit_kmeans(customer:pd.DataFrame, X_scaled):
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    customer['Cluster'] = km.fit_predict(X_scaled)
    
    plt.figure(figsize=(10,6))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(5):
        cluster_data = customer[customer["Cluster"] == i ]
        plt.scatter(cluster_data["Annual Income (k$)"], 
                    cluster_data["Spending Score (1-100)"],
                    c=colors[i], label=f"cluster{i}" , s=80)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments — K-Means (k=5)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/clusters.png")
    plt.show()
    return customer
        
    