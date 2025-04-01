import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import math


def load_data():
    # Load the marketing campaign dataset
    print("Loading data...")
    marketing_df = pd.read_csv('../data/digital_marketing_campaign_dataset.csv')
    return marketing_df


def perform_exploratory_analysis(marketing_df):
    print("Performing exploratory analysis...")
    # Checking for missing values
    print("Checking missing values...")
    print(marketing_df.isnull().sum())

    # Display summary statistics
    print("Summary statistics:")
    print(marketing_df.describe())

    # Check data types and column names
    print("Data types and column names:")
    print(marketing_df.dtypes)
    print(marketing_df.columns)

    # Display unique values for specific columns
    print("Unique values in 'CampaignChannel':")
    print(marketing_df['CampaignChannel'].unique())
    print("Unique values in 'CampaignType':")
    print(marketing_df['CampaignType'].unique())


def create_campaign_segment(marketing_df):
    print("Creating campaign segments...")
    def segment_by_campaign(row):
        return f"{row['CampaignType']} - {row['CampaignChannel']}"

    marketing_df['CampaignSegment'] = marketing_df.apply(segment_by_campaign, axis=1)
    print("Campaign segments created.")
    return marketing_df


def correlation_analysis(marketing_df):
    print("Performing correlation analysis...")
    # List of columns to exclude
    exclude_cols = ['CustomerID', 'CampaignChannel', 'CampaignType', 'CampaignSegment']

    # Separate columns into numerical and categorical (excluding the ones in exclude_cols)
    numerical_cols = marketing_df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
    categorical_cols = marketing_df.select_dtypes(exclude=[np.number]).columns.difference(exclude_cols)

    # One-hot encode categorical columns
    marketing_df_encoded = pd.get_dummies(marketing_df, columns=categorical_cols)

    # Select the relevant columns for correlation (including one-hot encoded categorical features)
    # Exclude original categorical columns to avoid the error
    columns_to_use = marketing_df_encoded.select_dtypes(include=[np.number]).columns.difference(exclude_cols)

    # List of group-by columns
    group_by_cols = ['CampaignType', 'CampaignChannel', 'CampaignSegment']

    # Loop over each group column and calculate correlation with other variables
    for group_col in group_by_cols:
        # Ensure group_col is numerical or one-hot encoded and calculate correlation
        corr_matrix = marketing_df_encoded[columns_to_use].corrwith(marketing_df_encoded[group_col].astype('category').cat.codes)

        # Print out the correlation results for each group column
        print(f"Correlation with {group_col}:")
        print(corr_matrix.sort_values(ascending=False))
        print("\n" + "="*50 + "\n")


def grouped_analysis(marketing_df, group_by_cols):
    print("Performing grouped analysis...")
    numeric_columns = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit', 
                       'TimeOnSite', 'SocialShares', 'PreviousPurchases', 'LoyaltyPoints', 'Conversion']  

    # Use all the group_by_cols provided
    group_by_cols_encoded = group_by_cols  

    adspend_dfs = {}
    cpa_dfs = {}
    conversion_dfs = {}

    for group_col in group_by_cols_encoded:
        if group_col not in marketing_df.columns:
            print(f"Warning: {group_col} not found in marketing_df columns")
            continue
        
        df_grouped = marketing_df.groupby(group_col)[numeric_columns].sum().reset_index()

        adspend_dfs[group_col] = df_grouped[[group_col, 'AdSpend']].sort_values(by='AdSpend', ascending=False)
        df_grouped_conversion = marketing_df.groupby(group_col)['Conversion'].mean().reset_index()
        conversion_dfs[group_col] = df_grouped_conversion.sort_values(by='Conversion', ascending=False)

        df_grouped[f'{group_col}CPA'] = df_grouped['AdSpend'] / df_grouped['Conversion']
        cpa_dfs[group_col] = df_grouped[[group_col, f'{group_col}CPA']].sort_values(by=f'{group_col}CPA', ascending=False)

    return adspend_dfs, cpa_dfs, conversion_dfs


def plot_metric(df, group_col, metric_col, title, ylabel, save_path):
    df_sorted = df.sort_values(by=metric_col, ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=group_col, y=metric_col, data=df_sorted)
    plt.title(title, fontsize=16)
    plt.xlabel(group_col, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the plot after saving


def plot_all_metrics(final_dfs, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for group_col, metrics in final_dfs.items():
        plot_metric(metrics['AdSpend'], group_col, 'AdSpend', f"AdSpend by {group_col}", 'Ad Spend ($)', os.path.join(save_folder, f"{group_col}_AdSpend.png"))
        plot_metric(metrics['Conversion'], group_col, 'Conversion', f"Conversion Rate by {group_col}", 'Conversion Rate', os.path.join(save_folder, f"{group_col}_Conversion.png"))
        plot_metric(metrics['CPA'], group_col, f'{group_col}CPA', f"CPA by {group_col}", 'Cost Per Acquisition ($)', os.path.join(save_folder, f"{group_col}_CPA.png"))

def plot_countplots(marketing_df, categorical_features, save_folder):
    print("Generating countplots...")

    num_features = len(categorical_features)
    num_cols = 1  # Fixed number of columns
    num_rows = num_features  # Each feature gets its own row

    plt.figure(figsize=(10, 6 * num_rows))

    for i, feature in enumerate(categorical_features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(x='CustomerSegment', hue=feature, data=marketing_df)
        plt.title(f"Distribution of {feature} Across Segments")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_folder, "categorical_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Countplots saved to {save_path}")


def plot_boxplots(marketing_df, numerical_features, save_folder):
    print("Generating boxplots...")

    num_features = len(numerical_features)
    num_cols = 3  # Fixed number of columns for better layout
    num_rows = math.ceil(num_features / num_cols)

    plt.figure(figsize=(12, 6 * num_rows))

    for i, feature in enumerate(numerical_features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(x='CustomerSegment', y=feature, data=marketing_df)
        plt.title(f"Comparison of {feature} Across Segments")

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_folder, "numerical_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Boxplots saved to {save_path}")


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
    bank_revenue_df = pd.read_csv('../data/bankrevenue.csv')
        
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


def main():
    # Load the marketing data
    marketing_df = load_data()

    # Perform exploratory analysis
    perform_exploratory_analysis(marketing_df)

    # Create campaign segments
    marketing_df = create_campaign_segment(marketing_df)

    # Perform correlation analysis
    correlation_analysis(marketing_df)

    # Grouped analysis by CampaignType and CampaignChannel
    group_by_cols = ['CampaignType', 'CampaignChannel', 'CampaignSegment']
    adspend_dfs, cpa_dfs, conversion_dfs = grouped_analysis(marketing_df, group_by_cols)

    # Plot all metrics and save to 'visualizations' folder
    save_folder = 'visualizations'
    final_dfs = {
        'CampaignType': {'AdSpend': adspend_dfs['CampaignType'], 
                         'Conversion': conversion_dfs['CampaignType'], 
                         'CPA': cpa_dfs['CampaignType']},

        'CampaignChannel': {'AdSpend': adspend_dfs['CampaignChannel'], 
                            'Conversion': conversion_dfs['CampaignChannel'], 
                            'CPA': cpa_dfs['CampaignChannel']},

        'CampaignSegment': {'AdSpend': adspend_dfs['CampaignSegment'], 
                            'Conversion': conversion_dfs['CampaignSegment'], 
                            'CPA': cpa_dfs['CampaignSegment']}
    }
    plot_all_metrics(final_dfs, save_folder)

    # Perform clustering (returns updated marketing_df with CustomerSegment)
    marketing_df, best_k, best_score, categorical_features, numerical_features = perform_clustering(marketing_df)
    print(f"Best number of clusters: {best_k}")
    print(f"Best score: {best_score}")


    # Create cluster profiles before merging bank revenue
    numerical_profiles = marketing_df.groupby('CustomerSegment')[[ 
        'Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
        'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
        'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints'
    ]].mean()

    categorical_profiles = marketing_df.groupby('CustomerSegment')[[ 
        'Gender', 'CampaignType', 'CampaignChannel', 'CampaignSegment'
    ]].agg(lambda x: x.mode()[0])

    cluster_profiles = pd.concat([numerical_profiles, categorical_profiles], axis=1).reset_index()

    # Merge bank revenue data
    cluster_profiles = merge_bank_revenue(marketing_df, cluster_profiles) 
    print(cluster_profiles)

    # Rank personalization
    cluster_profiles = rank_personalization(cluster_profiles)

    # Display the final results
    print(cluster_profiles.head())
    
    plot_countplots(marketing_df, categorical_features, save_folder)
    plot_boxplots(marketing_df, numerical_features, save_folder)



if __name__ == "__main__":
    main()
