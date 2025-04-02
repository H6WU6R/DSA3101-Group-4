
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data():
    df = pd.read_csv("./data/raw/digital_marketing_campaign_dataset.csv")
    df_drop = df.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID'])
    return df_drop

def model1_engagement_to_conversion(df_drop):
    engagement_columns = [ "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares", "EmailOpens", "EmailClicks" ]

    X = df_drop[engagement_columns]
    y = df_drop["Conversion"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("=== Predicting Conversion from Engagement ===")
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    importance_dict = dict(zip(engagement_columns, rf.feature_importances_))
    importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    print("\nRaw Feature Importances (importance_dict):")
    for feature, importance in importance_dict.items():
        print(f"{feature}: {importance:.4f}")


    weights = {key: value / sum(importance_dict.values()) for key, value in importance_dict.items()}
    print(weights)

    df_drop[engagement_columns] = scaler.fit_transform(df_drop[engagement_columns])
    df_drop["EngagementScore"] = sum(df_drop[col] * weights[col] for col in engagement_columns)


    return df_drop, weights

def model2_nonengagement_to_engagementscore(df_drop):
    engagement_columns = [ "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares", "EmailOpens", "EmailClicks" ]

    non_engagement_features = [
        col for col in df_drop.columns
        if col not in engagement_columns + ['CustomerID', 'ConversionRate', 'Conversion', 'AdvertisingPlatform', 'AdvertisingTool', 'EngagementScore']
    ]
    print(non_engagement_features)

    X = df_drop[non_engagement_features].copy()
    y = df_drop["EngagementScore"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_cols = ['Gender', 'CampaignChannel', 'CampaignType']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ])

    pipe_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)
    print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred_rf):.4f}")

    rf_model = pipe_rf.named_steps["model"]
    rf_features = numerical_cols + list(pipe_rf.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_cols))
    rf_importances = pd.Series(rf_model.feature_importances_, index=rf_features)
    print("\nTop Feature Importances (Random Forest):")
    sorted_rf = rf_importances.sort_values(ascending=False)
    for feature, importance in sorted_rf.head(10).items():
            print(f"{feature}: {importance:.4f}")

    pipe_gb = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    pipe_gb.fit(X_train, y_train)
    y_pred_gb = pipe_gb.predict(X_test)
    print(f"Gradient Boosting MSE: {mean_squared_error(y_test, y_pred_gb):.4f}")

    gb_model = pipe_gb.named_steps["model"]
    gb_importances = pd.Series(gb_model.feature_importances_, index=rf_features)

    # Print top 10 GBR importances
    print("\nTop Feature Importances (Gradient Boosting):")
    sorted_gb = gb_importances.sort_values(ascending=False)
    for feature, importance in sorted_gb.head(10).items():
        print(f"{feature}: {importance:.4f}")

def main():
    df_drop = load_data()
    df_drop, weights = model1_engagement_to_conversion(df_drop)
    model2_nonengagement_to_engagementscore(df_drop)

if __name__ == "__main__":
    main()
