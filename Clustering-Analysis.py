import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = "C:/Users/msi/Desktop/wine-clustering.csv"
df = pd.read_csv(file_path)
numeric_columns = df.select_dtypes(include=[np.number]).columns
X = df[numeric_columns].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)



kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X_pca)
y_kmeans = kmeans.predict(X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')


centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering with PCA')


plt.show()




import scipy.cluster.hierarchy as sch

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X_pca, method='ward'))
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
