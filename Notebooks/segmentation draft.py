from sklearn.preprocessing import StandardScaler
import pandas as pd

####################################################################################################
### Objective of model: Segment Customers Based On Their Features      ###         
####################################################################################################


def process_data(df_original):
    """
    Processes the dataset by:
    1. Dropping unnecessary columns
    2. Encoding categorical features using one-hot encoding
    3. Standardizing numerical features

    Returns:
    - df_scaled (numpy array): Standardized dataset
    - df_encoded (DataFrame): Encoded dataset before scaling (for column reference)
    - scaler (StandardScaler): Fitted scaler for inverse transformation
    """
    # Drop unwanted columns
    df_drop = df_original.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID'])

    # Encode categorical variables
    columns_to_encode = ['Gender', 'CampaignChannel', 'CampaignType']
    df_encoded = pd.get_dummies(df_drop, columns=columns_to_encode, drop_first=False)

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)

    return df_scaled, df_encoded, scaler


def find_best_pca_kmeans(df_scaled, max_pca=10, k_min=2, k_max=15):
    """
    Finds the optimal number of PCA components and K-Means clusters 
    using silhouette scores as the evaluation metric.

    Parameters:
    - df_scaled (numpy array): Standardized dataset
    - max_pca (int): Maximum number of PCA components to consider (default: 10)
    - k_min (int): Minimum number of clusters to test (default: 2)
    - k_max (int): Maximum number of clusters to test (default: 15)

    Returns:
    - best_pca_n (int): Best number of PCA components
    - best_k_for_best (int): Best number of clusters
    - best_scores_for_best (list): Silhouette scores for different k-values
    """
    # Define PCA and KMeans search ranges
    pca_range = range(2, min(df_scaled.shape[1], max_pca) + 1)
    k_range = range(k_min, k_max)
    
    overall_best_sil = -np.inf
    best_pca_n = None
    best_k_for_best = None
    best_scores_for_best = None

    # Loop over different numbers of PCA components
    for n_components in pca_range:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca_temp = pca.fit_transform(df_scaled)
        
        scores_temp = []
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_pca_temp)
            sil = silhouette_score(X_pca_temp, labels_temp)
            scores_temp.append((k, sil))

        best_for_this = max(scores_temp, key=lambda x: x[1])

        if best_for_this[1] > overall_best_sil:
            overall_best_sil = best_for_this[1]
            best_pca_n = n_components
            best_k_for_best = best_for_this[0]
            best_scores_for_best = scores_temp

    return best_pca_n, best_k_for_best, best_scores_for_best


def apply_pca_kmeans(df_scaled, best_pca_n, best_k_for_best):
    """
    Applies PCA with the optimal number of components and performs K-Means clustering.

    Parameters:
    - df_scaled (numpy array): Standardized dataset
    - best_pca_n (int): Best number of PCA components
    - best_k_for_best (int): Best number of clusters

    Returns:
    - df_pca (numpy array): PCA-transformed dataset
    - labels (numpy array): Cluster labels from K-Means
    - kmeans (KMeans object): Fitted K-Means model
    - pca_best (PCA object): Fitted PCA model
    """
    pca_best = PCA(n_components=best_pca_n, random_state=42)
    df_pca = pca_best.fit_transform(df_scaled)

    kmeans = KMeans(n_clusters=best_k_for_best, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca)

    return df_pca, labels, kmeans, pca_best


def track_kmeans_centroids(kmeans, pca_best, scalar, df_encoded, best_k_for_best):
    """
    Tracks the centroids of K-Means clusters and transforms them back to the original feature space.

    Parameters:
    - kmeans (KMeans object): Fitted K-Means model
    - pca_best (PCA object): Fitted PCA model used for dimensionality reduction
    - scalar (StandardScaler object): Fitted scaler used for standardization
    - df_encoded (DataFrame): The encoded dataframe before scaling
    - best_k_for_best (int): The optimal number of clusters

    Returns:
    - centroids_df (DataFrame): DataFrame of centroids mapped back to the original feature space
    """

    # 1. Get centroids in PCA-transformed space
    pca_centroids = kmeans.cluster_centers_

    # 2. Inverse transform using the correct PCA model
    centroids_standardized = pca_best.inverse_transform(pca_centroids)

    # 3. Reverse standardization to map back to original feature space
    centroids_original = centroids_standardized * scalar.scale_ + scalar.mean_

    # 4. Create DataFrame for better readability
    centroids_df = pd.DataFrame(centroids_original, columns=df_encoded.columns)

    # 5. Add cluster labels (0, 1, ..., k-1)
    centroids_df.insert(0, 'Cluster', range(best_k_for_best))

    # 6. Display results
    print("Final Cluster Centroids in Original Feature Space:")
    print(centroids_df.round(2))

    return centroids_df