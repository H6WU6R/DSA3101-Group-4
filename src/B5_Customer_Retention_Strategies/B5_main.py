import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def data_preprocessing(churn_path, seg_path):
    # Load data
    churn_data = pd.read_csv(churn_path)
    seg_data = pd.read_csv(seg_path)

    # Data cleaning
    churn_data = churn_data.drop(['RowNumber', 'CustomerId', 'Surname', 'Complain'], axis=1)
    seg_data = seg_data[['Age', 'Gender', 'Income', 'Cluster_Label']]

    # Classify income interval
    bins = np.arange(0, 210000, 10000)
    churn_data.loc[:, 'Income_bin'] = pd.cut(churn_data['EstimatedSalary'], bins=bins, right=False)
    seg_data.loc[:, 'Income_bin'] = pd.cut(seg_data['Income'], bins=bins, right=False)

    # Splitting into train and test sets
    churn_train, churn_test = train_test_split(churn_data, random_state=42, stratify=churn_data["Exited"],test_size=0.3)
    seg_train, seg_test = train_test_split(seg_data, random_state=42,test_size=0.3)

    # Merging data
    train= pd.merge(churn_train,
                    seg_train,
                    left_on=['Age', 'Gender','Income_bin'],
                    right_on=['Age', 'Gender','Income_bin'],
                    how='inner'
                )
    train=train.drop_duplicates()

    test=pd.merge(churn_test,
                    seg_test,
                    left_on=['Age', 'Gender',"Income_bin"],
                    right_on=['Age', 'Gender',"Income_bin"],
                    how='inner'
                )
    test=test.drop_duplicates()
    print("Merging customer churn dataset with segmentation dataset...")
    
    # Spliting label
    X_train=train.drop(["Exited", "Income_bin"], axis=1)
    y_train=train["Exited"].astype("category")

    X_test=test.drop(["Exited","Income_bin"],axis=1)
    y_test=test["Exited"].astype("category")

    # Encoding categorical variables
    le = LabelEncoder()
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = le.fit_transform(X_train[col])
    for col in X_test.select_dtypes(include=['object']).columns:
        X_test[col] = le.fit_transform(X_test[col])

    # Using SMOTE to solve imbalanced data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Using SMOTE to solve the problem of imbalanced data...this may take long")
    
    return X_train, X_test, y_train, y_test


def model_training(X_train, y_train):
    models = {}

    # Logistic Regression
    scaler = StandardScaler()
    X_train_lr=scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=500)
    
    lr.fit(X_train_lr, y_train)
    
    models['Logistic Regression'] = lr

    # Random Forest with Grid Search
    rf = RandomForestClassifier(criterion='entropy', max_depth=7, min_samples_leaf=2,
                       n_estimators=500, random_state=42)
    
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # Gradient Boosting with Random Search
    gb = GradientBoostingClassifier(learning_rate=0.3, max_depth=8, min_samples_leaf=10,
                           n_estimators=600, random_state=42)
    
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb

    return models, scaler


def model_evaluation(models, X_test, y_test, scaler):
    results = []
    for name, model in models.items():
        if name=='Logistic Regression':
            X_test_lr = scaler.transform(X_test)
            y_pred = model.predict(X_test_lr)     
        else: y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append((name, accuracy, roc_auc,f1))
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'ROC-AUC','F1 Score'])
    print(results_df)
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"Best model for customer churn prediction: {best_model['Model']} with accuracy: {best_model['Accuracy']}")


def main():
    X_train, X_test, y_train, y_test = data_preprocessing('./data/raw/Customer-Churn-Records.csv', './src/A1_Customer_Segmentation/A1-segmented_df.csv')
    models, scaler = model_training(X_train, y_train)
    model_evaluation(models, X_test, y_test, scaler)


if __name__ == '__main__':
    main()

