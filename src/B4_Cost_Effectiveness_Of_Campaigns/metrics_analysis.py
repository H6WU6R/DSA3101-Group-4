import pandas as pd

def calculate_cpa(df):
    """Calculate CPA (Cost Per Acquisition) for each group."""
    df['CPA'] = df['AdSpend'] / df['Conversion']
    print(f"CPA calculated. First few rows:\n{df[['AdSpend', 'Conversion', 'CPA']].head()}")
    return df

def calculate_roi(df):
    """Calculate ROI (Return On Investment) for each group."""
    df['ROI'] = (df['Rev_Total'] - df['CPA']) / df['CPA']
    print(f"ROI calculated. First few rows:\n{df[['Rev_Total', 'CPA', 'ROI']].head()}")
    return df

def group_by_campaign_and_calculate_metrics(df, group_by_cols):
    """Group by columns and calculate AdSpend, Conversion, and CPA."""
    final_dfs = {}
    for group_col in group_by_cols:
        print(f"Grouping by {group_col} and calculating metrics.")
        df_grouped = df.groupby(group_col).agg({
            'AdSpend': 'sum',
            'Conversion': 'mean'
        }).reset_index()

        df_grouped = calculate_cpa(df_grouped)
        final_dfs[group_col] = df_grouped
        print(f"Grouped DataFrame for {group_col}:\n{df_grouped.head()}")

    return final_dfs
