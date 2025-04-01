import numpy as np
from scipy.sparse import csr_matrix, identity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from sklearn.model_selection import train_test_split, KFold
import itertools
import os
import pandas as pd

def main():
    """  
        Computes the top-3 accuracy for a recommender system model.
        This metric evaluates how often the true purchased items for each user 
        appear in the model's top-3 predicted recommendations.
        Users with no recorded purchases are excluded from accuracy computation.

        Business Rationale:
        -------------------
        - Reflects realistic scenarios: In real-world recommendation systems, users interact with only a small subset of items.
        - Focuses on meaningful users: Only evaluates users who made actual purchases — avoids inflating performance with inactive users.
        - Top-3 is practical: Many businesses (e.g., e-commerce, fintech) care about whether their top few suggestions convert.
        - Scalable scoring: Gives partial credit (e.g., 0.5) when only some of the true purchases are recommended, reflecting degrees of success.
        """
    def compute_top3_accuracy_for_fold(model, X_val, interactions_val, item_features, k=3): 
        num_users = X_val.shape[0]
        top3_acc = []
        users_with_purchases = 0  # Count users with at least one purchase (ground truth signal)

        for user_id in range(num_users):
            # Predict scores for all items for the current user
            scores = model.predict(user_id, np.arange(interactions_val.shape[1]),
                                user_features=X_val,
                                item_features=item_features)

            # Get top-k (e.g., top-3) recommended item indices
            top3_indices = np.argsort(-scores)[:k]

            # Identify the indices of items the user actually purchased
            true_positives = set(np.where(interactions_val[user_id].toarray().flatten() == 1)[0])
            n_true = len(true_positives)

            # Rule: Skip users with no purchases (no signal to validate against)
            if n_true == 0:
                continue

            users_with_purchases += 1  # Track valid users for evaluation

            # Rule: User purchased 3 or more items — normalize by top-3
            if n_true >= 3:
                count = len(set(top3_indices).intersection(true_positives))
                top3_acc.append(count / 3.0)

            # Rule: User purchased exactly 2 items, if both appear in top 3 items, give full 
            # credits, if one is correctly predicted, give half credit, else 0
            elif n_true == 2:
                intersection = set(top3_indices).intersection(true_positives)
                if len(intersection) == 2:
                    top3_acc.append(1.0)
                elif len(intersection) == 1:
                    top3_acc.append(0.5)
                else:
                    top3_acc.append(0.0)

            # Rule: User purchased exactly 1 item, give full credit only if it is predicted to 
            # be the top 1 recommendation, give half credit if it is predicted to be 2nd or 3rd 
            # place, else 0
            elif n_true == 1:
                if top3_indices[0] in true_positives:
                    top3_acc.append(1.0)
                elif len(set(top3_indices[1:]).intersection(true_positives)) > 0:
                    top3_acc.append(0.5)
                else:
                    top3_acc.append(0.0)

        # Edge Case: No users had purchases
        if users_with_purchases == 0:
            return 0.0

        # Final average top-3 accuracy across all valid users
        return np.mean(top3_acc)

    # Conduct GridSearch for the lighfm model
    """
        Performs grid search with cross-validation over LightFM hyperparameters.
        
        Uses precision@3 and a custom top-3 accuracy to evaluate model performance
        across different configurations.

        Parameters:
        - feature_list: list of selected user features to include
        - X_train_full: full user feature matrix (pandas DataFrame)
        - Y_train_bin: binary interaction matrix (DataFrame where 1 = purchased, 0 = not purchased)

        Returns:
        - best_params: hyperparameters with the highest top-3 accuracy
        - grid_results: list of all evaluated hyperparameter combinations with scores

        User feature scaling + sparse matrix: Prepares data for LightFM’s matrix factorization format.
        Class balancing: Ensures that underrepresented products get learned properly.
        Multiple metrics: Combines industry-standard (precision@k) with a custom business-aligned metric (top3_accuracy).
        Cross-validation: Ensures generalizability and prevents overfitting on one data split.
        Hyperparameter tuning: Helps find the most effective configuration for embedding learning.
        """

    def grid_search_cv(feature_list, X_train_full, Y_train_bin):
        # STEP 1: Extract & scale user features
        X_train_features = X_train_full[feature_list].copy()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        user_features = csr_matrix(X_train_scaled)  # Convert to sparse matrix for LightFM

        # STEP 2: Build sparse interaction matrix from Y_train_bin
        interactions = csr_matrix(Y_train_bin.values)
        num_items = interactions.shape[1]

        # STEP 3: Identity matrix for item features (used as one-hot encodings)
        item_features = identity(num_items, format='csr')

        # STEP 4: Define hyperparameter grid to explore
        param_grid = {
            'loss': ['warp', 'bpr', 'logistic'],          # Different learning objectives
            'no_components': [16, 32, 64],                # Embedding dimension size
            'learning_rate': [0.001, 0.01, 0.05],         # Step size during gradient descent
            'epochs': [30, 50],                           # Number of training iterations
            'user_alpha': [1e-5, 1e-4],                   # Regularization strength for users
            'item_alpha': [1e-5, 1e-4]                    # Regularization strength for items
        }

        # STEP 5: Set up 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_results = []
        upsample_factor = 2  # Used to emphasize minority class during training

        # STEP 6: Loop over all hyperparameter combinations
        for loss, no_components, learning_rate, epochs, user_alpha, item_alpha in itertools.product(
            param_grid['loss'],
            param_grid['no_components'],
            param_grid['learning_rate'],
            param_grid['epochs'],
            param_grid['user_alpha'],
            param_grid['item_alpha']
        ):
            fold_top3_acc = []
            fold_prec = []

            # STEP 7: Loop over each cross-validation fold
            for train_idx, val_idx in kf.split(user_features):
                X_train_cv = user_features[train_idx]
                X_val_cv = user_features[val_idx]

                # STEP 8: Prepare training interaction matrix for this fold
                fold_train = interactions[train_idx].toarray().astype(float)

                # STEP 9: Heuristic upsampling to balance positive samples per item
                # Rationale: addresses class imbalance — some products may have very few purchases
                for j in range(fold_train.shape[1]):
                    pos_count = np.sum(fold_train[:, j] == 1)
                    neg_count = np.sum(fold_train[:, j] == 0)
                    if pos_count / fold_train.shape[0] < 0.3:
                        fold_train[:, j] = np.where(fold_train[:, j] == 1,
                                                    fold_train[:, j] * upsample_factor,
                                                    fold_train[:, j])
                    elif neg_count / fold_train.shape[0] < 0.3:
                        fold_train[:, j] = np.where(fold_train[:, j] == 0,
                                                    fold_train[:, j] * upsample_factor,
                                                    fold_train[:, j])

                fold_train_sparse = csr_matrix(fold_train)

                # STEP 10: Prepare validation interactions (kept untouched)
                fold_val = interactions[val_idx].toarray()
                fold_val_sparse = csr_matrix(fold_val)

                # STEP 11: Train LightFM model on this fold
                model_cv = LightFM(loss=loss, no_components=no_components,
                                learning_rate=learning_rate,
                                user_alpha=user_alpha,
                                item_alpha=item_alpha,
                                random_state=42)
                model_cv.fit(fold_train_sparse,
                            user_features=X_train_cv,
                            item_features=item_features,
                            epochs=epochs,
                            num_threads=4)

                # STEP 12: Evaluate with standard metric — precision@3
                prec = precision_at_k(model_cv, fold_val_sparse,
                                    user_features=X_val_cv,
                                    item_features=item_features,
                                    k=3).mean()

                # STEP 13: Custom top-3 accuracy (business-aligned metric)
                # Rationale: gives nuanced credit based on how many of top 3 match true purchases
                top3_acc = compute_top3_accuracy_for_fold(model_cv, X_val_cv, fold_val_sparse, item_features, k=3)

                fold_top3_acc.append(top3_acc)
                fold_prec.append(prec)

            # STEP 14: Average across folds
            avg_top3_acc = np.mean(fold_top3_acc)
            avg_prec = np.mean(fold_prec)

            # STEP 15: Store results for this hyperparameter combo
            grid_results.append({
                'loss': loss,
                'no_components': no_components,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'user_alpha': user_alpha,
                'item_alpha': item_alpha,
                'top3_accuracy': avg_top3_acc,
                'precision@3': avg_prec
            })

            # Optional: Print progress
            print(f"Params: loss={loss}, components={no_components}, "
                f"lr={learning_rate}, epochs={epochs}, user_alpha={user_alpha}, item_alpha={item_alpha} -> "
                f"Top3 Accuracy: {avg_top3_acc:.4f}, Precision@3: {avg_prec:.4f}")

        # STEP 16: Return best result based on top-3 accuracy
        best_params = max(grid_results, key=lambda x: x['top3_accuracy'])
        return best_params, grid_results

    # Load imputed data
    df_imputed_original = pd.read_csv("./data/processed/imputed_data_with_label.csv")

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
    feature_cols = ["current_age", "retirement_age", "birth_month", "gender", "yearly_income", "total_debt",
        "credit_score", "Credit_limit", "Debit_limit", "Debit (Prepaid)_limit", "Credit_expires", "Debit_expires",
        "Debit (Prepaid)_expires", "has_chip", "num_credit_cards"]


    # Binarize the ordinal labels: define a "positive" interaction if label >= 2.
    label_cols = ['Label_Rewards_Credit_Card', 'Label_Insurance_Solutions',
                           'Label_Digital_Financing', 'Label_Home_Improvement_Loan',
                           'Label_Auto_Vehicle_Financing', 'Label_Commodity_Investment_Services',
                           'Label_Travel_Rewards_Card', 'Label_Savings_Investment_Plans',
                           'Label_Wealth_Management_Savings','Label_Card_Upgrade','Label_Retention_Efforts']
    Y = df_imputed_original[label_cols].copy()
    Y_bin = (Y >= 2).astype(int)

    # Prepare user features for tuning.
    # Define two feature sets:
    base_features = feature_cols.copy()
    expanded_features = feature_cols + target_cols  # expanded: include spending categories

    # Split the data into train and test (test set remains unmodified).
    X = df_imputed_original.copy()  # full data
    X_train_full, X_test, Y_train_bin, Y_test_bin = train_test_split(X, Y_bin, test_size=0.2, random_state=42)

    # Uncomment and run the following for best params, see notebook and branch B1 for detailed results
    # print("\n--- Running grid search for feature set: base ---")
    # best_params_base, all_results_base = grid_search_cv(base_features, X_train_full, Y_train_bin)
    # print(f"Best params for base features: {best_params_base}")
    # print("\n--- Running grid search for feature set: expanded ---")
    # best_params_expand, all_results_expand = grid_search_cv(expanded_features, X_train_full, Y_train_bin)
    # print(f"Best params for expanded features: {best_params_expand}")

    print('best parameters for base features are {loss: warp, no_components: 32, learning_rate: 0.05, epochs: 30, user_alpha: 0.0001, item_alpha: 0.0001}\n')
    print('best parameters for expanded features are {loss: warp, no_components: 64, learning_rate: 0.05, epochs: 50, user_alpha: 0.0001, item_alpha: 0.0001}\n')

    best_params = {'loss': 'warp', 'no_components': 64, 'learning_rate': 0.05, 'epochs': 50, 'user_alpha': 0.0001, 'item_alpha': 0.0001}
    
    # Run the function below to test for test set data accuracy, it is not needed in the production-use but for 
    # fine tuning as more customer data is collected.
    def evaluate_lightfm_model(X_train_full, X_test, Y_train_bin, Y_test_bin, feature_set, best_params, label_cols):
    # 1. Feature Selection and Standardization
        X_train_selected = X_train_full[feature_set].copy()
        X_test_selected = X_test[feature_set].copy()
        
        scaler_final = StandardScaler()
        X_train_final = scaler_final.fit_transform(X_train_selected)
        X_test_final = scaler_final.transform(X_test_selected)
        
        # Convert to sparse matrices for efficiency
        user_features_train_final = csr_matrix(X_train_final)
        user_features_test_final = csr_matrix(X_test_final)

        # 2. Prepare Interaction Data
        final_interactions_train = csr_matrix(Y_train_bin.values)
        final_interactions_test = csr_matrix(Y_test_bin.values)
        num_items = len(label_cols)
        final_item_features = identity(num_items, format='csr')

        # 3. Train Final Model with Best Parameters
        model_final = LightFM(
            loss=best_params['loss'],
            no_components=best_params['no_components'],
            learning_rate=best_params['learning_rate'],
            user_alpha=best_params['user_alpha'],
            item_alpha=best_params['item_alpha'],
            random_state=42
        )

        model_final.fit(
            final_interactions_train,
            user_features=user_features_train_final,
            item_features=final_item_features,
            epochs=best_params['epochs'],
            num_threads=4
        )

        # 4. Evaluation Metrics
        # Precision@3
        final_precision = precision_at_k(
            model_final,
            final_interactions_test,
            user_features=user_features_test_final,
            item_features=final_item_features,
            k=3
        ).mean()

        # Custom Top-3 Accuracy
        custom_top3_accuracy = compute_top3_accuracy_for_fold(
            model_final, 
            user_features_test_final, 
            final_interactions_test, 
            final_item_features, 
            k=3
        )

        # 5. Generate Recommendations (Key Part)
        top3_recommendations = {}
        for user_id in range(user_features_test_final.shape[0]):
            # Predict scores for all items
            scores = model_final.predict(
                user_id, 
                np.arange(num_items),
                user_features=user_features_test_final,
                item_features=final_item_features
            )
            
            # Get indices of top 3 highest scores
            top3_indices = np.argsort(-scores)[:3]
            
            # Map indices to product names
            recommended_products = [label_cols[idx] for idx in top3_indices]
            top3_recommendations[user_id] = recommended_products

        return {
            'precision@3': final_precision,
            'custom_top3_accuracy': custom_top3_accuracy
        }, top3_recommendations
    
    # Expanded features shows better performance, use that as an example
    metrics, recommendations = evaluate_lightfm_model(X_train_full, X_test, Y_train_bin, Y_test_bin, 
                                                    expanded_features, best_params, label_cols)
    print("\n")
    print("---Test set results---")
    print("\nFinal Model Metrics (using expanded features):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nTop 3 product recommendations for sample test users:")
    for uid in list(recommendations.keys())[:5]:
        print(f"User {uid}: {recommendations[uid]}")


    # In real-life scenario, new customers do not have the label columns as they have not purchased any products
    # yet, hence we will run the funtion below to generate recommendations based on the model using the best parameters.

    def generate_lightfm_recommendations(X_train_full, X_test, Y_train_bin, feature_set, best_params, label_cols):
        """
        Trains a LightFM model using the best hyperparameters and generates top-3 product recommendations
        for each user in the test set. Evaluation metrics are not computed in this version.

        Parameters:
        - X_train_full: DataFrame of all training user features
        - X_test: DataFrame of test user features
        - Y_train_bin: DataFrame of binary interaction matrix for training
        - feature_set: List of selected user feature columns
        - best_params: Dict of tuned LightFM hyperparameters
        - label_cols: List of item/product labels (column names)

        Returns:
        - top3_recommendations: Dictionary mapping user_id → list of top-3 recommended products
        """
        
        # 1. Feature Selection and Standardization
        X_train_selected = X_train_full[feature_set].copy()
        X_test_selected = X_test[feature_set].copy()
        
        scaler_final = StandardScaler()
        X_train_final = scaler_final.fit_transform(X_train_selected)
        X_test_final = scaler_final.transform(X_test_selected)
        
        user_features_train_final = csr_matrix(X_train_final)
        user_features_test_final = csr_matrix(X_test_final)

        # 2. Prepare Interaction Data
        final_interactions_train = csr_matrix(Y_train_bin.values)
        num_items = len(label_cols)
        final_item_features = identity(num_items, format='csr')

        # 3. Train Final Model
        model_final = LightFM(
            loss=best_params['loss'],
            no_components=best_params['no_components'],
            learning_rate=best_params['learning_rate'],
            user_alpha=best_params['user_alpha'],
            item_alpha=best_params['item_alpha'],
            random_state=42
        )

        model_final.fit(
            final_interactions_train,
            user_features=user_features_train_final,
            item_features=final_item_features,
            epochs=best_params['epochs'],
            num_threads=4
        )

        # 4. Generate Top-3 Recommendations per User
        top3_recommendations = {}
        for user_id in range(user_features_test_final.shape[0]):
            scores = model_final.predict(
                user_id, 
                np.arange(num_items),
                user_features=user_features_test_final,
                item_features=final_item_features
            )
            top3_indices = np.argsort(-scores)[:3]
            recommended_products = [label_cols[idx] for idx in top3_indices]
            top3_recommendations[user_id] = recommended_products

        return top3_recommendations
    
    # Since we do not have new user data, we will be using test set data as example
    recommendations_for_new_users = generate_lightfm_recommendations(X_train_full, X_test, Y_train_bin,
                                                    expanded_features, best_params, label_cols)
    print("\n--- New User Recommendation results ---\n")
    cleaned_dict = {
    user_id: [label.replace('Label_', '') for label in labels]
    for user_id, labels in recommendations_for_new_users.items()}

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(cleaned_dict, orient='index')
    df.columns = ['Top1', 'Top2', 'Top3']
    print(df.iloc[5:10,:])
    # Save to CSV
    df.to_csv('src/B1_Predicting_Customer_Preferences/recommendations.csv', index_label='UserID')

    # In the feature engineering notebook provided, we have also experimented with PCA and Polynomial Features with PCA
    # However it is not used in the final code as the performance was not satisfying enough,
    # the coded provided is for future execution.

    X_expanded = df_imputed_original[expanded_features].copy()
    scaler_exp = StandardScaler()
    X_expanded_scaled = scaler_exp.fit_transform(X_expanded)

    # Assume X_expanded_scaled is already computed (using StandardScaler on your expanded features)
    n_total = X_expanded_scaled.shape[1] 

    # In our experiment, best number of component is 23, Cumulative explained variance = 0.8923

    """
    Uncomment and run the loop below for finding number of components with 90% of variance explained.
    for n in range(1, n_total + 1):
        pca_temp = PCA(n_components=n, random_state=42)
        pca_temp.fit(X_expanded_scaled)
        cum_explained = np.sum(pca_temp.explained_variance_ratio_)
        print(f"n_components = {n}: Cumulative explained variance = {cum_explained:.4f}")
    """

    pca_no_poly = PCA(n_components=23, random_state=42)  # Adjust n_components as needed.
    X_pca = pca_no_poly.fit_transform(X_expanded_scaled)
    pca_feature_names = [f'pca_no_poly_{i}' for i in range(X_pca.shape[1])]

    df_pca_no_poly = pd.DataFrame(X_pca, columns=pca_feature_names, index=df_imputed_original.index)

    final_feature_set = pca_feature_names

    # Split the PCA-transformed data and binarized labels into train and test sets.
    X_train_pca, X_test_pca, Y_train_bin, Y_test_bin = train_test_split(df_pca_no_poly, Y_bin, test_size=0.2, random_state=42)
    """
    Uncomment and run the code below for finding best parameters of the PCA model
    print("\n--- Running grid search for PCA features (no polynomial interactions) ---")
    best_params_pca, all_results_pca = grid_search_cv(final_feature_set, X_train_pca, Y_train_bin)
    print(f"Best PCA params: {best_params_pca}")
    """
    best_params_pca = {'loss': 'warp', 'no_components': 32, 'learning_rate': 0.05, 'epochs': 30, 'user_alpha': 0.0001, 'item_alpha': 0.0001}

    """
    Uncomment and run the code below to see the recommendation made using pca
    recommendation_pca = generate_lightfm_recommendations(X_train_pca, X_test_pca, Y_train_bin, 
    final_feature_set, best_params_pca, label_cols)

    print("\nFinal Model Metrics using PCA on expanded data (no polynomial interactions):")
    for k, v in metrics_pca.items():
        print(f"{k}: {v:.4f}")

    print("\nTop 3 product recommendations for sample test users using PCA (no polynomial interactions):")
    for uid in list(recommendations_pca.keys())[:5]:
        print(f"User {uid}: {recommendations_pca[uid]}")
    """

    """
    Refer to the feature_engineering notebook in branch B1 for the training results of Polynomial features"
    """
# Define a evaluation function to find 
if __name__ == "__main__":
    main()
