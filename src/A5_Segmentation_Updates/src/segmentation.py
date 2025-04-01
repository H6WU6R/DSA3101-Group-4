import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew
import pickle

global_scaler = None
global_model = None
global_numeric_columns = None
marketing_recs = {
    0: "Focus on digital marketing campaigns with emphasis on social media engagement.",
    1: "Develop loyalty programs and personalized email marketing strategies.",
    2: "Implement targeted promotional offers and cross-selling opportunities.",
    3: "Create brand awareness campaigns and introductory offers.",
    4: "Design retention programs with premium customer service support."
}

def preprocess_data(df_original, required_columns=None):
    """
    Preprocess the input DataFrame by:
      1. Dropping specified columns,
      2. Handling missing values,
      3. One-hot encoding categorical columns
    
    Returns:
      tuple: (df_encoded, None, None)
    """
    # Step 1: Drop specified columns
    df_drop = df_original.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID'])
    
    # Step 2: Handle missing values
    # For numeric columns, fill NaN with median
    numeric_columns = df_drop.select_dtypes(include=['int64', 'float64']).columns
    df_drop[numeric_columns] = df_drop[numeric_columns].fillna(df_drop[numeric_columns].median())
    
    # For categorical columns, fill NaN with mode (most frequent value)
    categorical_columns = ['Gender', 'CampaignChannel', 'CampaignType']
    for col in categorical_columns:
        df_drop[col] = df_drop[col].fillna(df_drop[col].mode()[0])
    
    # Step 3: One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_drop, columns=categorical_columns, drop_first=False)
    
    # Ensure the encoded dataframe has the same columns as the required columns
    if required_columns is not None:
        missing_cols = set(required_columns) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0
        # Ensure columns are in the same order
        df_encoded = df_encoded[required_columns]
    
    return df_encoded, None, None

def parse_contents(contents, filename):
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

def load_trained_models():
    """
    Load and return the saved KMeans model, PCA model, and scaler from pickle files.
    
    Returns:
      tuple: (kmeans_model, pca_model, scaler)
    """
    with open('data/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('data/pca_model.pkl', 'rb') as f:
        pca_model = pickle.load(f)
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    global global_model, global_scaler
    global_model = kmeans_model
    global_scaler = scaler
    return kmeans_model, pca_model, scaler

def predict_clusters(new_df, preprocess_fn=preprocess_data):
    """
    Preprocess new data, load the trained models, and predict cluster labels.
    """
    try:
        # Load saved models and scaler
        kmeans_model, pca_model, scaler = load_trained_models()
        
        # Get the required columns from the scaler
        required_columns = scaler.feature_names_in_
        
        # Make a copy of the input data to avoid modifications to original
        new_df_copy = new_df.copy()
        
        # Preprocess the new data
        df_encoded_new, _, _ = preprocess_fn(new_df_copy, required_columns=required_columns)
        
        # Use the loaded scaler to transform the new data
        scaled_new = scaler.transform(df_encoded_new)
        
        # Apply PCA transformation
        pca_new = pca_model.transform(scaled_new)
        
        # Predict clusters
        labels = kmeans_model.predict(pca_new)
        
        return labels
        
    except Exception as e:
        print(f"Error in predict_clusters: {str(e)}")
        raise

if __name__ == '__main__':
    # Test with sample data
    test_data = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Gender': ['M', 'F', 'M'],
        'CampaignChannel': ['Email', 'Social', 'Email'],
        'CampaignType': ['Awareness', 'Conversion', 'Awareness'],
        'AdvertisingPlatform': ['Facebook', 'Instagram', 'Twitter'],
        'AdvertisingTool': ['Video', 'Image', 'Text'],
        # Add other required columns with sample data
    })
    
    print("Test data shape:", test_data.shape)
    labels = predict_clusters(test_data)
    print("Predicted clusters:", labels)
    print("Number of predictions:", len(labels))