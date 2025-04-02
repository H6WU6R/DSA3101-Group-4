import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score

def save_cluster_distributions(data, features, save_dir="src/B3_Campaign_ROI_Evaluation"):
    """To save the clustering and distribution results."""

    os.makedirs(save_dir, exist_ok=True)

    # Saving distribution of clusters
    plt.figure(figsize=(10, 6))
    data['cluster'].value_counts().sort_index().plot(
        kind='bar',
        color='skyblue',
        edgecolor='black'
    )
    plt.title('Cluster Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.savefig(f"{save_dir}/cluster_distribution.png", bbox_inches='tight')
    plt.close()

    # Saving statistical distribution of features by cluster
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Distribution by Cluster', y=1.02)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        data.boxplot(
            column=feature,
            by='cluster',
            ax=axes[i],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red')
        )
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel('')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_distributions.png", bbox_inches='tight')
    plt.close()

def preprocess_data(data):
    """Load and preprocess raw data."""
    
    # Initial preprocessing
    data = data.drop(["Product_SKU", "Product_Description", "Product_Category", "Delivery_Charges"], axis=1)
    data["Amount"] = data["Avg_Price"] * data["Quantity"]
    data = data.drop(["Quantity", "Avg_Price"], axis=1)
    
    # Handle duplicates
    if data['Transaction_ID'].duplicated().any():
        data = data.drop_duplicates(subset='Transaction_ID', keep='first')
    
    # Convert dates
    data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'])
    return data

def engineer_features(grouped_data, reference_date):
    """Calculate RFMT metrics and purchase cycle."""
    # Calculate temporal features
    grouped_data['recency'] = (pd.to_datetime(reference_date) - grouped_data['last_purchase_date']).dt.days
    grouped_data['T'] = (pd.to_datetime(reference_date) - grouped_data['join_date']).dt.days
    
    # Calculate purchase cycle
    def calculate_purchase_cycle(dates):
        if len(dates) <= 1: return 0
        return sum((dates[i] - dates[i-1]).days for i in range(1, len(dates))) / (len(dates)-1)
    
    grouped_data['purchase_cycle'] = grouped_data['purchase_dates'].apply(calculate_purchase_cycle)
    return grouped_data.drop(columns=['purchase_dates', 'join_date', 'last_purchase_date'])

def prepare_final_dataset(data, split_date='2019-07-01'):
    """Create train/test split and calculate CLV targets."""
    # Temporal split
    train_data = data[data['Transaction_Date'] < split_date]
    test_data = data[data['Transaction_Date'] >= split_date]

    # Train features
    train_grouped = train_data.groupby('CustomerID').agg(
        join_date=('Transaction_Date', 'min'),
        last_purchase_date=('Transaction_Date', 'max'),
        frequency=('Transaction_ID', 'count'),
        monetary_value=('Amount', 'mean'),
        total_amount=('Amount', 'sum'),
        purchase_dates=('Transaction_Date', sorted),
        coupon_used=('Coupon_Status', lambda x: (x == 'True').sum())
    ).reset_index()

    train_features = engineer_features(train_grouped, split_date)
    
    # Test target
    test_target = test_data.groupby('CustomerID')['Amount'].sum().reset_index(name='actual_3m_CLV')
    
    # Merge datasets
    final_data = pd.merge(train_features, test_target, on='CustomerID', how='left').fillna(0)
    return final_data

def optimize_clusters(X):
    """Determine optimal number of clusters using elbow method and silhouette scores."""
    wcss = []
    silhouette_scores = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(K_range, wcss, marker='o')
    ax1.set_title('Elbow Method')
    ax1.set(xlabel='Number of Clusters', ylabel='WCSS')
    
    ax2.plot(K_range, silhouette_scores, marker='o', color='r')
    ax2.set_title('Silhouette Scores')
    ax2.set(xlabel='Number of Clusters', ylabel='Score')
    
    return K_range[np.argmax(silhouette_scores)]

def main():
    # Data pipeline
    df_original = pd.read_csv('./data/raw/Online_Sales.csv')
    data = preprocess_data(df_original)
    final_data = prepare_final_dataset(data)
    
    # Train/test split
    Customer_train, Customer_test = train_test_split(final_data, test_size=0.2, random_state=42)
    
    # Feature scaling
    features = ['recency', 'frequency', 'monetary_value', 'T', 'purchase_cycle']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(Customer_train[features])
    X_test = scaler.transform(Customer_test[features])
    
    # Cluster optimization
    best_k = optimize_clusters(X_train)
    
    # Model training
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    Customer_train['cluster'] = kmeans.fit_predict(X_train)

    # Clustering results
    final_data = final_data.merge(
        Customer_train[['CustomerID', 'cluster']],
        on='CustomerID',
        how='left'
    )
    plot_features = features + ['actual_3m_CLV']
    
    # Saving clustering results
    save_cluster_distributions(
        data=final_data,
        features=plot_features,
        save_dir='src/B3_Campaign_ROI_Evaluation'
    )
    
    # Prediction
    cluster_means = Customer_train.groupby('cluster')['actual_3m_CLV'].mean().to_dict()
    Customer_test['predicted_cluster'] = kmeans.predict(X_test)
    Customer_test['predicted_3m_CLV'] = Customer_test['predicted_cluster'].map(cluster_means)
    
if __name__ == "__main__":
    main()
