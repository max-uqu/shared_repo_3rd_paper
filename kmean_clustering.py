import pandas as pd

from sklearn.cluster import KMeans
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def cluster_assets(dataset: pd.DataFrame) -> pd.DataFrame:
    elbow_method_is_enabled = False

    if elbow_method_is_enabled:
        # Define the range of K values to test
        K_range = range(1, 11)  # You can adjust this range based on your dataset

        # Initialize an empty list to store the WCSS values for each K
        wcss = []

        # Compute WCSS for each value of K
        for K in K_range:
            kmeans = KMeans(n_clusters=K, init="k-means++", random_state=42)
            kmeans.fit(dataset)
            wcss.append(kmeans.inertia_)  # Inertia is the WCSS for the K-means model

        # Plot the Elbow Graph
        plt.figure(figsize=(8, 6))
        plt.plot(K_range, wcss, "bo-", markersize=8)
        plt.title("Elbow Method to Determine Optimal K", fontsize=16)
        plt.xlabel("Number of clusters (K)", fontsize=14)
        plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=14)
        plt.xticks(K_range)
        plt.grid(True)
        plt.show()

    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    dataset["cluster"] = kmeans.fit_predict(dataset)
    dataset["cluster"] = dataset["cluster"] + 1

    return dataset
