import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle



def preprocess_data(df_original):
    """
    Preprocess the input DataFrame by:
    1. Dropping specified columns
    2. One-hot encoding categorical columns
    3. Scaling all features
    
    Parameters:
    df_original (pd.DataFrame): The original DataFrame to process
    
    Returns:
    pd.DataFrame: The preprocessed and scaled DataFrame
    """
    # Step 1: Drop specified columns
    df_drop = df_original.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID'])
    
    # Step 2: One-hot encode categorical columns
    columns_to_encode = ['Gender', 'CampaignChannel', 'CampaignType']
    df_encoded = pd.get_dummies(df_drop, columns=columns_to_encode, drop_first=False)
    df_encoded.to_csv('./data/processed/A1-processed-df.csv', index=False)
    # Step 3: Scale all features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    
    # Convert back to DataFrame (since StandardScaler returns numpy array)
    df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns)
    
    return df_encoded, df_scaled, scaler


def optimize_clusters(df_scaled):
    """
    Find optimal PCA components and K-Means clusters by maximizing silhouette score.
    
    Parameters:
    df_scaled (pd.DataFrame): Preprocessed and scaled DataFrame
    
    Returns:
    tuple: (best_pca_n, best_k, best_silhouette, df_pca, labels)
        - best_pca_n: Optimal number of PCA components
        - best_k: Optimal number of clusters
        - best_silhouette: Best silhouette score achieved
        - df_pca: DataFrame after PCA transformation
        - labels: Cluster labels from best K-Means model
    """
    # Define parameter ranges
    pca_range = range(2, min(df_scaled.shape[1], 10) + 1)  # From 2 to either 10 or number of features
    k_range = range(2, 15)  # Test cluster numbers from 2 to 14
    
    # Initialize variables to track best results
    overall_best_sil = -np.inf
    best_pca_n = None
    best_k = None
    best_scores = None

    print("Loop over different numbers of PCA components and cluster number...this may take long")
    # Loop over different numbers of PCA components
    for n_components in pca_range:
        # Apply PCA with n_components
        pca = PCA(n_components=n_components, random_state=42)
        X_pca_temp = pca.fit_transform(df_scaled)
        
        # Test K-Means for this PCA-transformed data
        scores_temp = []
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            labels_temp = kmeans_temp.fit_predict(X_pca_temp)
            sil = silhouette_score(X_pca_temp, labels_temp)
            scores_temp.append((k, sil))
        
        # Find the best silhouette score for this number of PCA components
        best_for_this = max(scores_temp, key=lambda x: x[1])
        
        # Check if this configuration is better than the overall best
        if best_for_this[1] > overall_best_sil:
            overall_best_sil = best_for_this[1]
            best_pca_n = n_components
            best_k = best_for_this[0]
            best_scores = scores_temp
    
    # Fit the best PCA and K-Means models
    pca_best = PCA(n_components=best_pca_n, random_state=42)
    df_pca = pca_best.fit_transform(df_scaled)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(df_pca)
    print("Best number of PCA components:", best_pca_n)
    print("Best number of clusters (k):", best_k)
    print("Overall best silhouette score:", overall_best_sil)
    print("First ten cluster labels:")
    print(labels[:10])  # Show the first 10 labels
    
    return best_pca_n, best_k, overall_best_sil, df_pca, labels,kmeans, pca_best



def get_cluster_centroids(pca_best, kmeans, scaler, df_encoded, best_k):
    """
    Calculate and return cluster centroids in original feature space.
    
    Parameters:
    pca_best (PCA): Fitted PCA model with optimal components
    kmeans (KMeans): Fitted KMeans model with optimal clusters
    scaler (StandardScaler): Fitted scaler used for preprocessing
    df_encoded (pd.DataFrame): Encoded DataFrame (before scaling)
    best_k (int): Optimal number of clusters
    
    Returns:
    pd.DataFrame: DataFrame containing cluster centroids in original feature space
    """
    # 1. Get centroids in PCA-transformed space
    pca_centroids = kmeans.cluster_centers_
    
    # 2. Inverse transform using the correct PCA model
    centroids_standardized = pca_best.inverse_transform(pca_centroids)
    
    # 3. Reverse standardization to map back to original feature space
    centroids_original = centroids_standardized * scaler.scale_ + scaler.mean_
    
    # 4. Create DataFrame for better readability
    centroids_df = pd.DataFrame(centroids_original, columns=df_encoded.columns)
    
    # 5. Add cluster labels (0, 1, ..., k-1)
    centroids_df.insert(0, 'Cluster', range(best_k))
    print("Centroids DataFrame:")
    print(centroids_df)

    return centroids_df
    
def main():
    # 1. Load data
    df_original = pd.read_csv('./data/raw/digital_marketing_campaign_dataset.csv')

    # 2. Preprocess data (need to modify preprocess_data to return 3 objects)
    df_encoded, df_scaled, scaler = preprocess_data(df_original)
    
    # 3. Optimize clusters (need to modify optimize_clusters to return 7 objects)
    best_pca_n, best_k, overall_best_sil, df_pca, labels,kmeans, pca_best = optimize_clusters(df_scaled)
    
    # 4. Get centroids
    centroids_df = get_cluster_centroids(pca_best, kmeans, scaler, df_encoded, best_k)
    
    df_original['Cluster_Label'] = labels
    # Save with relative path from main.py's location
    df_original.to_csv('./src/A1_Customer_Segmentation/A1-segmented_df.csv', index=False)

    # Save models with relative paths
    with open('./src/A1_Customer_Segmentation/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    with open('./src/A1_Customer_Segmentation/pca_model.pkl', 'wb') as f:
        pickle.dump(pca_best, f)
    with open('./src/A1_Customer_Segmentation/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return df_original

if __name__ == "__main__":
    main()
