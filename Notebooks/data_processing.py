import numpy as np
import pandas as pd

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
