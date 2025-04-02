import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(marketing_df):
    print("Performing clustering...")
    # Handle missing values
    marketing_df.fillna(marketing_df.median(numeric_only=True), inplace=True)

    # Select numerical features for clustering
    numerical_features = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints', 'Conversion']
    X = marketing_df[numerical_features]

    # Encode categorical features
    categorical_features = ['Gender', 'CampaignType', 'CampaignChannel', 'CampaignSegment']
    encoded_df = pd.get_dummies(marketing_df[categorical_features], drop_first=True)
    X = pd.concat([X, encoded_df], axis=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality Reduction using PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Determine the best number of clusters using silhouette score
    best_k = 0
    best_score = -1
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        if score > best_score:
            best_k = k
            best_score = score

    # Apply KMeans with the best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    marketing_df['CustomerSegment'] = kmeans.fit_predict(X_pca)
    return marketing_df, best_k, best_score, categorical_features, numerical_features


def merge_bank_revenue(marketing_df, cluster_profiles):
    print("Merging bank revenue data...")
    # Load bank revenue data
    bank_revenue_df = pd.read_csv('./data/raw/bankrevenue.csv')
        
    # Merge bank revenue data with marketing_df
    marketing_df = marketing_df.merge(bank_revenue_df[['AGE', 'Rev_Total']], left_on='Age', right_on='AGE', how='left')
    
    # Convert Rev_Total to numeric (in case there are string values)
    marketing_df['Rev_Total'] = marketing_df['Rev_Total'].replace({'\$': '', ',': ''}, regex=True)
    marketing_df['Rev_Total'] = pd.to_numeric(marketing_df['Rev_Total'])


    # Step 2: Calculate total revenue per CustomerSegment
    segment_revenue = marketing_df.groupby('CustomerSegment')['Rev_Total'].sum().reset_index()

    # Step 3: Merge total revenue data with cluster profiles
    cluster_profiles = cluster_profiles.merge(segment_revenue, on='CustomerSegment', how='left')

    # Ensure CPA exists and calculate if not already done
    if 'CPA' not in cluster_profiles.columns:
        if 'AdSpend' in cluster_profiles.columns and 'ConversionRate' in cluster_profiles.columns:
            cluster_profiles['CPA'] = cluster_profiles['AdSpend'] / cluster_profiles['ConversionRate']
        else:
            raise KeyError("CPA is missing, and AdSpend/ConversionRate are also missing. Cannot calculate CPA.")
    
    # Step 4: Compute ROI for each customer segment using CPA
    cluster_profiles['ROI'] = (cluster_profiles['Rev_Total'] - cluster_profiles['CPA']) / cluster_profiles['CPA']

    return cluster_profiles


def rank_personalization(cluster_profiles):
    print("Ranking personalization...")
    cluster_profiles['CPA_Level'] = pd.qcut(cluster_profiles['CPA'], 3, labels=['Low', 'Medium', 'High'])
    cluster_profiles['ROI_Level'] = pd.qcut(cluster_profiles['ROI'], 3, labels=['Low', 'Medium', 'High'])

    cluster_profiles['CPA_Points'] = cluster_profiles['CPA_Level'].map({'Low': 3, 'Medium': 2, 'High': 1}).astype(int)
    cluster_profiles['ROI_Points'] = cluster_profiles['ROI_Level'].map({'Low': 1, 'Medium': 2, 'High': 3}).astype(int)
    return cluster_profiles


