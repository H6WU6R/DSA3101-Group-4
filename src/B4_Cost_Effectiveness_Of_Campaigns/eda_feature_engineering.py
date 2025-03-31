import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def basic_eda(df):
    """Perform basic exploratory data analysis."""
    print("Performing basic EDA...")
    print("Missing values in the dataset:")
    print(df.isnull().sum())
    print("Summary statistics of the dataset:")
    print(df.describe())
    print("Data types of each column:")
    print(df.dtypes)
    print("Column names in the dataset:")
    print(df.columns)

def plot_correlation_matrix(df):
    """Plot the correlation matrix for numerical features."""
    print("Plotting correlation matrix for numerical features...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    print("Correlation matrix plotted.")

def create_campaign_segment(df):
    """Create a campaign segment column by combining CampaignType and CampaignChannel."""
    print("Creating campaign segment column...")
    def segment_by_campaign(row):
        return f"{row['CampaignType']} - {row['CampaignChannel']}"
    df['CampaignSegment'] = df.apply(segment_by_campaign, axis=1)
    print("Campaign segment column created.")
    return df
