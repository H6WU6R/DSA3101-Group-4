import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 

def main():
    # Load finalized csv files
    final = pd.read_csv("../data/processed/B1/final_data.csv")

    # For customers without spending transaction data, use KNNImputer to compute its missing values.
    # We start with finding the best K, you may refer to branch B1 for detailed approach to find 
    # best K with elbow method, all numeric values need to be standardized to be used for KNN method.
    def evaluate_knn_imputer_scaled_transform_original(n_neighbors, data, target_cols, mask_frac=0.1, random_state=42):
        mse_list = []
        # Work on a copy so that modifications for one column do not affect the others.
        data_copy = data.copy()
        np.random.seed(random_state)
        
        for col in target_cols:
            non_missing_idx = data_copy[data_copy[col].notnull()].index
            if len(non_missing_idx) == 0:
                continue
            
            # Randomly mask a fraction of non-missing values
            mask_idx = np.random.choice(non_missing_idx, 
                                        size=int(mask_frac * len(non_missing_idx)), 
                                        replace=False)
            original_values = data_copy.loc[mask_idx, col].copy()
            data_copy.loc[mask_idx, col] = np.nan
            
            # Build a pipeline: Standardize then impute
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', KNNImputer(n_neighbors=n_neighbors))
            ])
            
            imputed_array = pipeline.fit_transform(data_copy)
            imputed_df = pd.DataFrame(imputed_array, columns=data_copy.columns, index=data_copy.index)
            
            # Retrieve the fitted scaler to manually scale original values
            fitted_scaler = pipeline.named_steps['scaler']
            col_index = list(data_copy.columns).index(col)
            
            # Scale the original (masked) values using the fitted scaler parameters
            original_values_scaled = (original_values - fitted_scaler.mean_[col_index]) / fitted_scaler.scale_[col_index]
            imputed_values_scaled = imputed_df.loc[mask_idx, col]
            
            mse = mean_squared_error(original_values_scaled, imputed_values_scaled)
            mse_list.append(mse)
            
            # Restore original values so the next iteration works on unmodified data_copy
            data_copy.loc[mask_idx, col] = original_values
            
        return np.mean(mse_list)

    # List of target columns of imputation
    target_cols = ['Automotive & Transportation Services', 'Clothing & Fashion',
        'Digital Goods & Computers', 'Electronics & Appliances',
        'Freight & Trucking', 'Hotels & Accommodation',
        'Legal & Financial Services', 'Machinery & Tools',
        'Medical & Healthcare Services', 'Movies & Theaters',
        'Postal Services - Government Only', 'Rail & Bus Transport',
        'Restaurants & Eating Places', 'Retail Stores',
        'Sports & Recreational Activities', 'Steel & Metal Products',
        'Telecommunications & Media', 'Utilities & Home Services']

    # List of feature columns used for imputation
    feature_cols = ['current_age', 'retirement_age', 'birth_month', 'gender',
        'latitude', 'longitude', 'yearly_income', 'total_debt', 'credit_score',
        'num_credit_cards', 'Credit', 'Debit', 'Debit (Prepaid)']

    # Create a working DataFrame with both feature and target columns.
    # (Assuming you have a DataFrame called 'final' already loaded.)
    cols_for_impute = feature_cols + target_cols
    df_impute = final[cols_for_impute].copy()

    print("Best n is found to be 5 using elbow method.\n")
    # Select the best n_neighbors (with the lowest MSE)
    best_n = 5

    # Build final pipeline and impute missing values using the best n_neighbors.
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=best_n))
    ])
    imputed_array = pipeline_final.fit_transform(df_impute)

    # Save the imputed & scaled DataFrame.
    df_imputed_scaled = pd.DataFrame(imputed_array, columns=df_impute.columns, index=df_impute.index)

    # Convert all values back to original scale for label construction.
    scaler = pipeline_final.named_steps['scaler']

    # Inverse-transform the imputed scaled data.
    df_imputed_original_array = scaler.inverse_transform(df_imputed_scaled)
    df_imputed_original = pd.DataFrame(df_imputed_original_array, columns=df_impute.columns, index=df_impute.index)
    print("Data imputation is done.\n")

    # Label Construction
    # Joint features are considered for label construction, while purchase behavior (e.g., spending in categories like 
    # restaurants, utilities, or digital goods) gives insight into interests and lifestyle, it doesn't tell the full 
    # story about a customer's financial capacity, needs, or creditworthiness — all of which are critical for 
    # financial product targeting.
    # Median and quantiles for threshold-based rules
    income_median = df_imputed_original['yearly_income'].median()
    num_cards_median = df_imputed_original['num_credit_cards'].median()
    credit_upper = df_imputed_original['credit_score'].quantile(0.75)
    income_80 = df_imputed_original['yearly_income'].quantile(0.80)

    # Define relevant spending categories and compute their 75th percentile values
    spending_categories = [
        'Retail Stores', 'Restaurants & Eating Places', 'Clothing & Fashion', 
        'Movies & Theaters', 'Sports & Recreational Activities', 'Freight & Trucking', 
        'Medical & Healthcare Services', 'Postal Services - Government Only', 
        'Digital Goods & Computers', 'Telecommunications & Media', 'Utilities & Home Services', 
        'Automotive & Transportation Services', 'Steel & Metal Products', 'Machinery & Tools', 
        'Rail & Bus Transport', 'Hotels & Accommodation', 'Legal & Financial Services'
    ]
    spending_upper = {cat: df_imputed_original[cat].quantile(0.75) for cat in spending_categories}

    # 1. Rewards Credit Card:
    #    Conditions: (a) any spending category (Retail, Restaurants, Fashion, Movies, Sports) above its upper quantile,
    #                (b) credit score above upper quantile,
    #                (c) number of credit cards above or equal to median.
    def label_rewards_credit_card(row):
        score = 0
        cats = ['Retail Stores', 'Restaurants & Eating Places', 'Clothing & Fashion', 
                'Movies & Theaters', 'Sports & Recreational Activities']
        if any(row[cat] >= spending_upper[cat] for cat in cats):  # (a)
            score += 1
        if row['credit_score'] >= credit_upper:  # (b)
            score += 1
        if row['num_credit_cards'] >= num_cards_median:  # (c)
            score += 1
        return min(score, 3)

    # 2. Insurance Solutions:
    #    Conditions: (a) any spending category (Freight, Healthcare, Government Postal) above its upper quantile,
    #                (b) credit score between median and 80th percentile,
    #                (c) yearly income above or equal to median.
    def label_insurance_solutions(row):
        score = 0
        cats = ['Freight & Trucking', 'Medical & Healthcare Services', 'Postal Services - Government Only']
        if any(row[cat] >= spending_upper[cat] for cat in cats):  # (a)
            score += 1
        credit_median = df_imputed_original['credit_score'].quantile(0.50)
        credit_80 = df_imputed_original['credit_score'].quantile(0.80)
        if credit_median < row['credit_score'] <= credit_80:  # (b)
            score += 1
        if row['yearly_income'] >= income_median:  # (c)
            score += 1
        return min(score, 3)

    # 3. Digital Financing:
    #    Conditions: (a) any spending category (Digital Goods or Telecom & Media) above its upper quantile,
    #                (b) yearly income above or equal to median,
    #                (c) credit score above upper quantile.
    def label_digital_financing(row):
        score = 0
        cats = ['Digital Goods & Computers', 'Telecommunications & Media']
        if any(row[cat] >= spending_upper[cat] for cat in cats):  # (a)
            score += 1
        if row['yearly_income'] >= income_median:  # (b)
            score += 1
        if row['credit_score'] >= credit_upper:  # (c)
            score += 1
        return min(score, 3)

    # 4. Home Improvement Loan:
    #    Conditions: (a) credit score must exceed upper quantile (mandatory to be eligible),
    #                (b) spending in Utilities & Home Services above its upper quantile,
    #                (c) yearly income above or equal to median.
    def label_home_improvement_loan(row):
        score = 0 
        if row['credit_score'] >= credit_upper:  # (a)
            score += 1
        if row['Utilities & Home Services'] >= spending_upper['Utilities & Home Services']:  # (b)
            score += 1
        if row['yearly_income'] >= income_median:  # (c)
            score += 1
        return min(score, 3)

    # 5. Auto & Vehicle Financing:
    #    Conditions: (a) credit score must exceed upper quantile (mandatory to be eligible),
    #                (b) spending in Automotive & Transportation Services above its upper quantile,
    #                (c) yearly income above or equal to median.
    def label_auto_vehicle_financing(row):
        score = 0
        if row['credit_score'] >= credit_upper:  # (a)
            score += 1
        if row['Automotive & Transportation Services'] >= spending_upper['Automotive & Transportation Services']:  # (b)
            score += 1
        if row['yearly_income'] >= income_median:  # (c)
            score += 1
        return min(score, 3)

    # 6. Commodity & Investment Services:
    #    Conditions: (a) any spending category (Steel & Metal Products or Machinery & Tools) above its upper quantile,
    #                (b) yearly income above 80th percentile,
    #                (c) low total debt relative to income (total_debt < 0.5 * yearly_income).
    def label_commodity_investment_services(row):
        score = 0
        cats = ['Steel & Metal Products', 'Machinery & Tools']
        if any(row[cat] >= spending_upper[cat] for cat in cats):  # (a)
            score += 1
        if row['yearly_income'] >= income_80:  # (b)
            score += 1
        if row['total_debt'] <= 0.5 * row['yearly_income']:  # (c)
            score += 1
        return min(score, 3)

    # 7. Travel Rewards Card:
    #    Conditions: (a) any spending category (Rail & Bus Transport or Hotels & Accommodation) above its upper quantile,
    #                (b) yearly income above or equal to median,
    #                (c) number of credit cards above or equal to median.
    def label_travel_rewards_card(row):
        score = 0
        cats = ['Rail & Bus Transport', 'Hotels & Accommodation']
        if any(row[cat] >= spending_upper[cat] for cat in cats):  # (a)
            score += 1
        if row['yearly_income'] >= income_median:  # (b)
            score += 1
        if row['num_credit_cards'] >= num_cards_median:  # (c)
            score += 1
        return min(score, 3)

    # 8. Savings/Investment Plans:
    #    Conditions: (a) total debit is at least 1.5x greater than credit,
    #                (b) low total debt relative to income (total_debt < 0.5 * yearly_income),
    #                (c) years until retirement <= 15.
    def label_savings_investment_plans(row):
        score = 0
        if row['Debit'] >= 1.5 * row['Credit']:  # (a)
            score += 1
        if row['total_debt'] < 0.5 * row['yearly_income']:  # (b)
            score += 1
        if row['retirement_age'] - row['current_age'] <= 15:  # (c)
            score += 1
        return min(score, 3)

    # 9. Wealth Management & Savings:
    #    Conditions: (a) spending in Legal & Financial Services above its upper quantile,
    #                (b) yearly income above 80th percentile and low total debt (total_debt < 0.5 * yearly_income),
    #                (c) credit score above upper quantile.
    def label_wealth_management_savings(row):
        score = 0
        if row['Legal & Financial Services'] >= spending_upper['Legal & Financial Services']:  # (a)
            score += 1
        if row['yearly_income'] >= income_80 and row['total_debt'] < 0.5 * row['yearly_income']:  # (b)
            score += 1
        if row['credit_score'] >= credit_upper:  # (c)
            score += 1
        return min(score, 3)

    # Apply label functions to construct new label columns.
    df_imputed_original['Label_Rewards_Credit_Card'] = df_imputed_original.apply(label_rewards_credit_card, axis=1)
    df_imputed_original['Label_Insurance_Solutions'] = df_imputed_original.apply(label_insurance_solutions, axis=1)
    df_imputed_original['Label_Digital_Financing'] = df_imputed_original.apply(label_digital_financing, axis=1)
    df_imputed_original['Label_Home_Improvement_Loan'] = df_imputed_original.apply(label_home_improvement_loan, axis=1)
    df_imputed_original['Label_Auto_Vehicle_Financing'] = df_imputed_original.apply(label_auto_vehicle_financing, axis=1)
    df_imputed_original['Label_Commodity_Investment_Services'] = df_imputed_original.apply(label_commodity_investment_services, axis=1)
    df_imputed_original['Label_Travel_Rewards_Card'] = df_imputed_original.apply(label_travel_rewards_card, axis=1)
    df_imputed_original['Label_Savings_Investment_Plans'] = df_imputed_original.apply(label_savings_investment_plans, axis=1)
    df_imputed_original['Label_Wealth_Management_Savings'] = df_imputed_original.apply(label_wealth_management_savings, axis=1)

    # Save data to processed folder
    #save_path = os.path.join(BASE_DIR, 'data/processed/B1/imputed_data_with_label.csv')

    # Uncomment the line to save the file
    # df_imputed_original.to_csv(save_path, index=False)
    print("Label construction is done.\n")

if __name__ == "__main__":
    main()