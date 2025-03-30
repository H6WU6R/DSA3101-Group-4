import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, boxcox, yeojohnson
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore 



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
    
    return centroids_df


def main():
    # 1. Load data
    df_original = pd.read_csv('/Users/cindy/Desktop/DSA3101-Project-3/Data/digital_marketing_campaign_dataset.csv')

    # 2. Preprocess data (need to modify preprocess_data to return 3 objects)
    df_encoded, df_scaled, scaler = preprocess_data(df_original)
    
    # 3. Optimize clusters (need to modify optimize_clusters to return 7 objects)
    best_pca_n, best_k, overall_best_sil, df_pca, labels,kmeans, pca_best = optimize_clusters(df_scaled)
    
    # 4. Get centroids
    centroids_df = get_cluster_centroids(pca_best, kmeans, scaler, df_encoded, best_k)
    
    df_original['Cluster_Label'] = labels
    return df_original


