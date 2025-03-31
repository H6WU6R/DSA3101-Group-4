from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def apply_pca(X_scaled, n_components=3):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA applied with {n_components} components. Explained variance ratio: {pca.explained_variance_ratio_}")
    return X_pca

def optimal_k_for_clustering(X_pca):
    """Find the optimal number of clusters using silhouette score."""
    max_k = min(len(X_pca) - 1, 10)
    print(f"Finding optimal k. Maximum k is set to: {max_k}")
    
    best_k = None
    best_score = -1
    
    for k in range(2, max_k + 1):
        print(f"Testing k = {k}")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_pca)
        labels = kmeans.labels_
        score = silhouette_score(X_pca, labels)
        print(f"Silhouette score for k = {k}: {score}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Best k found: {best_k} with silhouette score: {best_score}")
    return best_k, best_score

def perform_clustering(X_pca, best_k):
    """Perform K-Means clustering with the optimal number of clusters."""
    print(f"Performing KMeans clustering with k = {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    print(f"Clustering complete. Cluster centers: {kmeans.cluster_centers_}")
    return labels
