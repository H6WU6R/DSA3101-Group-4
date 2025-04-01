"""
Data processing module for customer behavioral data.

This module contains functions for loading, cleaning and preparing customer data
for segmentation analysis.
"""

import pandas as pd
import numpy as np
import os


def behavioral_data_processing(file_path=None):
    """
    Process customer behavioral data by applying segmentation and engineering additional
    behavioral metrics.
    
    This function:
    1. Loads customer data from the specified CSV file
    2. Cleans the data by removing unnecessary columns and converting data types
    3. Engineers new behavioral features including:
       - Email Click-Through Rate (CTR)
       - Website Engagement Depth
       - Social Sharing Propensity
    
    Args:
        file_path (str, optional): Path to the CSV file containing customer data.
            Defaults to '/Users/cindy/Desktop/DSA3101-Project-3/Data/A1-segmented_df.csv'.
        
    Returns:
        pd.DataFrame: Processed DataFrame with original columns plus engineered features
    
    """
    
    if file_path is None:
        file_path = '/Users/cindy/Desktop/DSA3101-Project-3/Data/A1-segmented_df.csv'
    
    df = pd.read_csv(file_path)
    
    # Data cleaning
    # 1. Remove unnecessary columns if they exist
    columns_to_drop = ['AdvertisingPlatform', 'AdvertisingTool']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 2. Convert categorical columns to category dtype for efficiency
    categorical_cols = ['Gender', 'CampaignChannel', 'CampaignType']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 3. Feature engineering: Create additional behavioral metrics
    
    # Email Click-Through Rate (CTR)
    if 'EmailClicks' in df.columns and 'EmailOpens' in df.columns:
        df['email_ctr'] = df.apply(
            lambda row: row['EmailClicks'] / row['EmailOpens'] if row['EmailOpens'] > 0 else 0, 
            axis=1
        )
    
    # Website Engagement Depth
    if 'PagesPerVisit' in df.columns and 'TimeOnSite' in df.columns:
        df['engagement_depth'] = df['PagesPerVisit'] * df['TimeOnSite']
    
    # Social Sharing Propensity
    if 'SocialShares' in df.columns and 'WebsiteVisits' in df.columns:
        df['social_propensity'] = df.apply(
            lambda row: row['SocialShares'] / row['WebsiteVisits'] if row['WebsiteVisits'] > 0 else 0,
            axis=1
        )
    
    return df
