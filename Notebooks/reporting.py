import pandas as pd
import numpy as np

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


