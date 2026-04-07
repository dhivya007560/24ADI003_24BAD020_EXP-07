print("DHIVYA A : 24BAD020")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv("E:\\ASSIGNMENTS\\ML\\DATASETS\\Mall_Customers.csv")
if data.isnull().sum().any():
    data = data.dropna()
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
aic = []
bic = []
K_range = range(1, 11)
for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    aic.append(gmm.aic(X_scaled))
    bic.append(gmm.bic(X_scaled))
plt.figure()
plt.plot(K_range, aic, marker='o', label='AIC')
plt.plot(K_range, bic, marker='s', label='BIC')
plt.title("AIC & BIC for GMM")
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.legend()
plt.show()
optimal_k = 5
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm.fit(X_scaled)
probabilities = gmm.predict_proba(X_scaled)
clusters = np.argmax(probabilities, axis=1)
data['GMM_Cluster'] = clusters


log_likelihood = gmm.score(X_scaled)
sil_score = silhouette_score(X_scaled, clusters)
print(f"\nLog-Likelihood: {log_likelihood}")
print(f"Silhouette Score: {sil_score}")
print(f"AIC: {gmm.aic(X_scaled)}")
print(f"BIC: {gmm.bic(X_scaled)}")


plt.figure()
plt.hist(probabilities.max(axis=1), bins=20)
plt.title("Cluster Probability Distribution")
plt.xlabel("Max Probability")
plt.ylabel("Frequency")
plt.show()


plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
x = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100)
X_grid, Y_grid = np.meshgrid(x, y)
grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
Z = -gmm.score_samples(grid)
Z = Z.reshape(X_grid.shape)
plt.contour(X_grid, Y_grid, Z)
plt.title("GMM Contour Plot")
plt.xlabel("Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.show()


kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_scaled)
plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters)
plt.title("K-Means Clustering")

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.title("GMM Clustering")
plt.show()
print("\nCluster Means (GMM):")
print(data.groupby('GMM_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

print("\nSample Data:")
print(data.head())
