#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, chi2_contingency
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# Data Loading Module
# ---------------------------
def load_data(file_path):
    print(f"Loading data from: {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df

# ---------------------------
# Engagement Score Module (with Scaling)
# ---------------------------
def compute_engagement_score(df, kpi_columns, test_size=0.3, random_state=42):
    """
    Trains a logistic regression model using the specified KPI columns to predict conversion,
    computes a continuous engagement score for each observation, scales it to a 0–10 range,
    and adds both the raw and scaled scores to the DataFrame.
    Returns the modified DataFrame, the trained logistic regression model, and the scaler.
    """
    print("Computing engagement score via logistic regression...")
    
    # Split data for model training
    train_df, _ = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Conversion'])
    X_train = train_df[kpi_columns].fillna(0)
    y_train = train_df['Conversion']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_train_scaled, y_train)
    
    # Compute raw engagement score for the entire dataset
    X_all = df[kpi_columns].fillna(0)
    X_all_scaled = scaler.transform(X_all)
    coefficients = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    df['engagement_score'] = np.dot(X_all_scaled, coefficients) + intercept
    
    # Scale the engagement score to a 0–10 range
    scaler_engagement = MinMaxScaler(feature_range=(0, 10))
    df['scaled_engagement_score'] = scaler_engagement.fit_transform(df[['engagement_score']])
    
    print("Engagement score computed and scaled (0-10).")
    return df, lr_model, scaler

# ---------------------------
# KPI Threshold Selection via ROC Analysis Module
# ---------------------------
def select_thresholds_via_roc(df, kpi_columns, test_size=0.3, random_state=42):
    """
    For each KPI, splits the data into training and testing sets, computes the optimal threshold
    using ROC curve analysis (maximizing Youden's J statistic) on the training data.
    Returns a DataFrame with the optimal threshold for each KPI.
    """
    # Split the data for threshold selection
    train_df, _ = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Conversion'])
    thresholds = {}
    
    for kpi in kpi_columns:
        y_true_train = train_df['Conversion'].values
        scores_train = train_df[kpi].values
        
        fpr, tpr, thresh = roc_curve(y_true_train, scores_train)
        J = tpr - fpr  # Youden's J statistic
        best_index = np.argmax(J)
        best_threshold = thresh[best_index]
        thresholds[kpi] = best_threshold
        
    thresholds_df = pd.DataFrame.from_dict(thresholds, orient='index', columns=["Best Threshold"])
    return thresholds_df

# ---------------------------
# Overall KPI Metrics Module using Optimal Thresholds
# ---------------------------
def compute_overall_kpi_metrics_with_optimal_thresholds(df, optimal_thresholds):
    """
    For each KPI, creates a binary indicator column (e.g., 'ClickThroughRate_met')
    that is 1 if the KPI meets/exceeds the optimal threshold.
    Computes KPI_hit_count as the sum of these binary indicators.
    """
    print("Computing overall KPI metrics using optimal thresholds...")
    for kpi, threshold in optimal_thresholds.items():
        df[kpi + '_met'] = (df[kpi] >= threshold).astype(int)
    met_columns = [kpi + '_met' for kpi in optimal_thresholds.keys()]
    df['KPI_hit_count'] = df[met_columns].sum(axis=1)
    print("Overall KPI metrics computed.")
    return df

# ---------------------------
# Main Production Workflow
# ---------------------------
def main():
    # File path (read-only; no local file modifications)
    main_data_path = "../Data/digital_marketing_campaign_dataset.csv"
    
    # 1. Load main dataset
    df = load_data(main_data_path)
    
    # 2. Define KPI columns (ensure these columns exist in your dataset)
    kpi_columns = [
        'ClickThroughRate',
        'ConversionRate',
        'WebsiteVisits',
        'PagesPerVisit',
        'TimeOnSite',
        'EmailOpens',
        'EmailClicks'
    ]
    
    # 3. Compute engagement score (raw) and scaled engagement score using logistic regression
    df, lr_model, scaler = compute_engagement_score(df, kpi_columns)
    
    # 4. Select optimal thresholds for each KPI using ROC analysis (Youden's J statistic)
    thresholds_df = select_thresholds_via_roc(df, kpi_columns)
    
    # 5. Compute overall KPI metrics (KPI_hit_count) using the optimal thresholds
    optimal_thresholds = thresholds_df["Best Threshold"].to_dict()
    df = compute_overall_kpi_metrics_with_optimal_thresholds(df, optimal_thresholds)
    
    print(thresholds_df)
    
    sample_output_columns = ['scaled_engagement_score', 'KPI_hit_count', 'Conversion']
    print("\nSample Production Output (first 20 rows):")
    print(df[sample_output_columns].head(10))
    
if __name__ == "__main__":
    main()

