# segmentation.py

import base64
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from scipy.stats import skew

# Global variables to hold our scaler and clustering model
global_scaler = None
global_model = None
global_numeric_columns = None

# Define marketing recommendations mapping (example text)
marketing_recs = {
    0: "High spenders: Target with premium offers and loyalty programs.",
    1: "Moderate spenders: Use personalized discount codes to boost engagement.",
    2: "Low spenders: Encourage increased usage with introductory promotions.",
    3: "Frequent cash advance users: Provide financial advice and repayment incentives.",
    4: "Occasional users: Use re-engagement campaigns to drive usage."
}

def fill_missing_and_engineer_features(df):
    """Handle missing values, drop unwanted columns, and create new features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            min_val = df[col].min()
            df[col].fillna(min_val, inplace=True)
    
    if 'CREDIT_LIMIT' in df.columns:
        df['BALANCE_UTILIZATION'] = df['BALANCE'] / df['CREDIT_LIMIT']
        df.drop(columns=['CREDIT_LIMIT'], inplace=True)
    
    if 'CUST_ID' in df.columns:
        df.drop(columns=['CUST_ID'], inplace=True)
    
    df['ONEOFF_PURCHASE_RATIO'] = np.where(df['PURCHASES'] != 0,
                                           df['ONEOFF_PURCHASES'] / df['PURCHASES'], 0)
    
    df['INSTALLMENT_PURCHASE_RATIO'] = np.where(df['PURCHASES'] != 0,
                                                df['INSTALLMENTS_PURCHASES'] / df['PURCHASES'], 0)
    
    df['ADVANCE_RATIO'] = np.where((df['PURCHASES'] + df['CASH_ADVANCE']) != 0,
                                   df['CASH_ADVANCE'] / (df['PURCHASES'] + df['CASH_ADVANCE']),
                                   0)
    
    df['PAYMENT_GAP'] = df['PAYMENTS'] - df['MINIMUM_PAYMENTS']
    df['PAYMENT_RATIO'] = np.where(df['MINIMUM_PAYMENTS'] != 0,
                                   df['PAYMENTS'] / df['MINIMUM_PAYMENTS'], 0)
    
    df['PURCHASES_PER_TRX'] = np.where(df['PURCHASES_TRX'] != 0,
                                       df['PURCHASES'] / df['PURCHASES_TRX'], 0)
    
    df['ADVANCE_PER_TRX'] = np.where(df['CASH_ADVANCE_TRX'] != 0,
                                     df['CASH_ADVANCE'] / df['CASH_ADVANCE_TRX'], 0)
    
    df['ONEOFF_PURCHASE_FREQ_RATIO'] = np.where(df['PURCHASES_FREQUENCY'] != 0,
                                                df['ONEOFF_PURCHASES_FREQUENCY'] / df['PURCHASES_FREQUENCY'], 0)
    
    df['INSTALLMENT_PURCHASE_FREQ_RATIO'] = np.where(df['PURCHASES_FREQUENCY'] != 0,
                                                     df['PURCHASES_INSTALLMENTS_FREQUENCY'] / df['PURCHASES_FREQUENCY'], 0)
    
    df['ADVANCE_FREQ_RATIO'] = np.where(df['PURCHASES_FREQUENCY'] != 0,
                                        df['CASH_ADVANCE_FREQUENCY'] / df['PURCHASES_FREQUENCY'], 0)
    return df

def apply_log_transform_selectively(df):
    """For each numeric column, if |skew| > 1 then apply log1p transform."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness_values = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    
    for col in numeric_cols:
        if abs(skewness_values[col]) > 1:
            if df[col].min() <= 0:
                shift_val = abs(df[col].min()) + 1e-5
                df[col] = df[col] + shift_val
            df[col] = np.log1p(df[col])
    
    new_skew = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    return df

def preprocess_data(df):
    """Apply full preprocessing: missing value handling, feature engineering, and log transform."""
    df = fill_missing_and_engineer_features(df)
    df = apply_log_transform_selectively(df)
    return df

def scale_data(df, scaler=None):
    """Scale numeric data using StandardScaler."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols]
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)
    return scaled, scaler, numeric_cols

def generate_cluster_summary(labels):
    """Generate a summary string for cluster assignment (if needed)."""
    return "Cluster assignment complete."

def generate_marketing_recommendations(labels):
    """Generate marketing recommendations based on predicted cluster."""
    # For simplicity, if multiple rows, show recommendation for each.
    recs = "Marketing Recommendations:\n"
    for i, label in enumerate(labels):
        rec = marketing_recs.get(label, "No recommendation available.")
        recs += f"Data row {i+1}: Cluster {label} -> {rec}\n"
    return recs

def parse_contents(contents, filename):
    """Parse uploaded file contents and return a DataFrame."""
    import base64, io
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        else:
            return None
    except Exception as e:
        print(e)
        return None

def initial_model_training():
    global global_scaler, global_model, global_numeric_columns, global_dataset
    initial_df = pd.read_csv('Data/Bank Customer Segmentation.csv')
    initial_df = preprocess_data(initial_df)
    initial_scaled, global_scaler, global_numeric_columns = scale_data(initial_df)
    initial_clusters = 5  # Default number of clusters
    from sklearn.cluster import MiniBatchKMeans
    global_model = MiniBatchKMeans(n_clusters=initial_clusters, random_state=42, batch_size=100)
    initial_labels = global_model.fit_predict(initial_scaled)
    global_dataset = initial_df  # Store dataset for later use
    return initial_df, initial_scaled, initial_labels


if __name__ == '__main__':
    # For testing the segmentation module independently
    initial_model_training()
