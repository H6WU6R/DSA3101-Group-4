import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimpy import skim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter
import warnings
from openpyxl import *
from datetime import timedelta
from lifetimes import BetaGeoFitter, GammaGammaFitter
warnings.filterwarnings("ignore")


def main():
    """
    1. Data Cleaning & Preprocessing
    """
    data = load_and_preprocess_data()

    """
    2. Calculate Customer Metrics
    """
    reference_date = pd.to_datetime('2020-01-01') 
    grouped = calculate_customer_metrics(data, reference_date)

    """
    3. Fit CLV Models
    """
    bgf, ggf = fit_clv_models(grouped)

    """
    4. Predict CLV
    """
    time_period = 365 
    clv_data = predict_clv(grouped, bgf, ggf, time_period)

    print(clv_data[['CustomerID', 'predicted_1yr_clv']])

def load_and_preprocess_data():
    data = pd.read_csv('../../data/processed/B3/Online_Sales.csv')
    data = data.drop(["Product_SKU", "Product_Description", "Product_Category", "Delivery_Charges", "Coupon_Status"], axis=1)
    data["Amount"] = data["Avg_Price"] * data["Quantity"]
    data = data.drop(["Quantity", "Avg_Price"], axis=1)
    
    customer = pd.read_excel('../../data/processed/B3/CustomersData.xlsx')
    data = pd.merge(data, customer, on="CustomerID", how='left')
    data = data.drop(["Gender", "Location"], axis=1)
    
    return data

def calculate_customer_metrics(data, reference_date):
    grouped = data.groupby('CustomerID').agg(
        join_date=('Transaction_Date', 'min'),
        last_purchase_date=('Transaction_Date', 'max'),
        frequency=('Transaction_ID', 'count'),
        monetary_value=('Amount', 'mean')
    ).reset_index()

    grouped['recency'] = (reference_date - grouped['last_purchase_date']).dt.days
    grouped['T'] = (reference_date - grouped['join_date']).dt.days
    
    return grouped

def fit_clv_models(grouped):
    bgf = BetaGeoFitter()
    bgf.fit(grouped["frequency"], grouped["recency"], grouped["T"])
    
    ggf = GammaGammaFitter()
    ggf.fit(grouped["frequency"], grouped["monetary_value"])
    
    return bgf, ggf

def predict_clv(grouped, bgf, ggf, time_period=365):
    grouped['predicted_transactions'] = bgf.predict(time_period, grouped['frequency'], grouped['recency'], grouped['T'])
    grouped['predicted_average_value'] = ggf.conditional_expected_average_profit(grouped['frequency'], grouped['monetary_value'])
    grouped['predicted_1yr_clv'] = grouped['predicted_transactions'] * grouped['predicted_average_value']
    
    return grouped

if __name__ == "__main__":
    main()



