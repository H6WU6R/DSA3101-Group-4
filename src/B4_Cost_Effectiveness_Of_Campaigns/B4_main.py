import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processing import load_data, handle_missing_values, select_features
from eda_feature_engineering import basic_eda, plot_correlation_matrix, create_campaign_segment
from clustering_pca import apply_pca, optimal_k_for_clustering, perform_clustering
from metrics_analysis import group_by_campaign_and_calculate_metrics
from visualisation import plot_metric, plot_cluster_profiles

def main():
    # Redirecting print statements to a file
    output_file = 'output_log.txt'  # Specify the output log file path
    sys.stdout = open(output_file, 'w')  # Redirect standard output to the file

    # Test to see if print statements work
    print("This should appear in the output file.")

    # Load data
    file_path = '../data/digital_marketing_campaign_dataset.csv'
    print(f"Loading data from {file_path}...")
    marketing_df = load_data(file_path)
    print(f"Data loaded. Shape of the dataset: {marketing_df.shape}")

    # Handle missing values
    marketing_df = handle_missing_values(marketing_df)
    print(f"Missing values handled. Shape of the dataset after handling missing values: {marketing_df.shape}")

    # Basic EDA
    basic_eda(marketing_df)
    plot_correlation_matrix(marketing_df)

    # Feature Engineering
    marketing_df = create_campaign_segment(marketing_df)
    print(f"CampaignSegment created. Shape of the dataset: {marketing_df.shape}")

    # Prepare data for clustering
    numerical_features = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'PreviousPurchases', 'LoyaltyPoints', 'Conversion']
    categorical_features = ['Gender', 'CampaignType', 'CampaignChannel', 'CampaignSegment']
    X = select_features(marketing_df, numerical_features, categorical_features)
    print(f"Features selected. Shape of the feature matrix X: {X.shape}")

    # Apply PCA and clustering
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = apply_pca(X_scaled)
    best_k, best_score = optimal_k_for_clustering(X_pca)
    marketing_df['CustomerSegment'] = perform_clustering(X_pca, best_k)

    # Calculate metrics
    group_by_cols = ['CampaignType', 'CampaignChannel', 'CampaignSegment']
    final_dfs = group_by_campaign_and_calculate_metrics(marketing_df, group_by_cols)

    # Plot metrics
    for group_col, df_final in final_dfs.items():
        print(f"Plotting metrics for group: {group_col}")
        plot_metric(df_final, group_col, 'AdSpend', f"AdSpend by {group_col}", 'Ad Spend ($)')
        plot_metric(df_final, group_col, 'Conversion', f"Conversion Rate by {group_col}", 'Conversion Rate')
        plot_metric(df_final, group_col, 'CPA', f"CPA by {group_col}", 'Cost Per Acquisition ($)')

    # Visualize cluster profiles
    categorical_features += ['CustomerSegment']
    plot_cluster_profiles(marketing_df, categorical_features, numerical_features)

    # Close the file after writing all the logs
    sys.stdout.close()

# Only run the main function if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()
