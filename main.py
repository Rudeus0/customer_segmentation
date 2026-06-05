from src.cluster import (data_load, preprocess, find_optimal_k, fit_kmeans, profile_cluster)

if __name__ == "__main__":

    customer = data_load()

    X_scaled, scaler = preprocess(customer)

    k_find = find_optimal_k()

    customer = fit_kmeans(customer, X_scaled)

    profile = profile_cluster(customer)

    print("\n--- Cluster Profiles (Means) ---")
    print(profile)