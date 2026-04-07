print("DHIVYA A : 24BAD020")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv("E:\\ASSIGNMENTS\\ML\\DATASETS\\Mall_Customers.csv")
if data.isnull().sum().any():
    data = data.dropna()
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method (K vs Inertia)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters
inertia_value = kmeans.inertia_
sil_score = silhouette_score(X_scaled, clusters)
print(f"\nInertia: {inertia_value}")
print(f"Silhouette Score: {sil_score}")

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, marker='X', label='Centroids')

plt.title("Customer Segmentation (K-Means)")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.show()
print("\nCluster-wise Mean Values:")
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
print("\nSample Data with Cluster Labels:")
print(data.head())

plt.figure()
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='X', s=300)

