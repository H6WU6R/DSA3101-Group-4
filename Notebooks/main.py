import os
import pandas as pd
from data_preprocessing import load_data, perform_exploratory_analysis, create_campaign_segment
from reporting import correlation_analysis, grouped_analysis
from visualisation import plot_all_metrics, plot_countplots, plot_boxplots
from clustering import perform_clustering, merge_bank_revenue, rank_personalization

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
