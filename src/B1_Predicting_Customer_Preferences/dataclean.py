import pandas as pd
import json
import csv
import os

# Load the datasets
# Please update the file paths according to your local setup
# Example:
# cards = pd.read_csv('path_to_your_file/cards_data.csv')
def main():
    print("Start to load datasets\n")
    cards = pd.read_csv('./data/raw/cards_data.csv')
    transactions_1 = pd.read_csv( './data/raw/transactions_part1.csv')
    transactions_2 = pd.read_csv('./data/raw/transactions_part2.csv')
    users = pd.read_csv( './data/raw/users_data.csv')
    print("All datasets are loaded successfully!\n")
    # Merge the transactions data
    transactions = pd.concat([transactions_1, transactions_2], ignore_index=True)
    print("Data Cleaning starts\n")
    # Clean the cards data
    # Convert 'credit_limit' to numeric and remove '$' sign
    cards['credit_limit']= cards['credit_limit'].replace({'\$': ''}, regex=True).astype(float)

    # Drop the sensitive columns.
    cards = cards.drop(columns=[
        'card_number', 
        'cvv', 
        'year_pin_last_changed', 
        'card_on_dark_web',
        'acct_open_date'
    ])

    # Extract the year from the 'expires' column (assuming format 'mm/yyyy').
    # Convert the extracted year to integer for proper comparison.
    cards['expires'] = cards['expires'].astype(str).str.split('/').str[1].astype(int)

    # Convert 'has_chip' to binary (1 for 'yes', 0 otherwise).
    cards['has_chip'] = cards['has_chip'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Group by 'client_id' and 'card_type' to aggregate:
    # - Sum of 'credit_limit'.
    # - The maximum 'expires' year as the latest expiration.
    grouped = cards.groupby(['client_id', 'card_type']).agg(
        total_credit_limit=('credit_limit', 'sum'),
        latest_expires=('expires', 'max')
    ).reset_index()

    # Pivot for total_credit_limit.
    pivot_credit = grouped.pivot(index='client_id', columns='card_type', values='total_credit_limit').fillna(0)
    pivot_credit.columns = [f"{col}_limit" for col in pivot_credit.columns]
    pivot_credit = pivot_credit.reset_index()

    # Pivot for latest_expires.
    pivot_expires = grouped.pivot(index='client_id', columns='card_type', values='latest_expires').fillna(0)
    pivot_expires.columns = [f"{col}_expires" for col in pivot_expires.columns]
    pivot_expires = pivot_expires.reset_index()

    # Merge the two pivoted DataFrames on client_id.
    cards_new = pivot_credit.merge(pivot_expires, on='client_id')

    # Since 'has_chip' and 'num_cards_issued' are client-level fields,
    # aggregate them separately by 'client_id'.
    # For 'has_chip', we'll use max (assuming if any card has a chip, we mark it as 1).
    # For 'num_cards_issued', we sum the values.
    client_info = cards.groupby('client_id').agg(
        has_chip=('has_chip', 'min'),
        num_cards_issued=('num_cards_issued', 'sum')
    ).reset_index()

    # Merge the pivoted card-type aggregates with the client-level info.
    cards_new = cards_new.merge(client_info, on='client_id')

    # Clean users data
    # Drop unnecessary columns
    # Convert 'yearly_income', 'total_debt' to numeric and remove '$' sign
    users.drop(columns=['address','per_capita_income','latitude','longitude'], inplace=True)
    users.loc[:,"yearly_income":"total_debt"]=users.loc[:,"yearly_income":"total_debt"].apply(lambda x: x.replace({'\$': ''}, regex=True).astype(float))

    # Convert 'gender' to binary, with 1 being Male, 0 being Female
    users['gender'] = users['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # First merge the users and cards data, using inner join on 'client_id' and 'id'
    final_df = pd.merge(users, cards_new, left_on='id', right_on='client_id', how='inner').sort_values('id').reset_index(drop=True)

    # Remove highly correlated columns :('id','client_id'), ('current_age', 'birth_year')
    final_df.drop(columns=['client_id','birth_year','num_cards_issued'], inplace=True)

    # Clean transactions data
    # Convert 'transaction_date' to datetime
    transactions["date"] = pd.to_datetime(transactions["date"])

    # Retrive the data from the latest year
    # Get the maximum date from the transactions DataFrame
    # This will give us the most recent transaction date
    end_date = transactions["date"].max()

    # Set the start date to be 1 year before the end date
    start_date = end_date - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    tran_df = transactions[(transactions["date"] >= start_date) & (transactions["date"] <= end_date)]

    # Remove erroneous transactions
    tran_df = tran_df[tran_df['errors'].isna()]
    tran_df.drop(columns=['id','date','card_id','merchant_id','merchant_city','merchant_state','zip'], inplace=True)
    tran_df.drop(columns=['errors'],inplace=True)
    tran_df['amount'] = tran_df['amount'].replace({'\$': ''}, regex=True).astype(float)

    # Load merchant code json file
    json_path = './data/raw/B1/mcc_codes.json'

    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group similar merchant codes into one category
    # Create a dictionary to map the merchant codes to their respective categories
    subgroup_mapping = {"Freight & Trucking": ["3730", "4214"],
        "Steel & Metal Products": ["3000", "3001", "3005", "3006", "3007", "3008", "3009",
                                    "3359", "3387", "3389", "3390", "3393", "3395", "3405", "5094"],
        "Retail Stores": ["5300", "5310", "5311", "5411", "5499","5193","3260"],
        "Digital Goods & Computers": ["5815", "5816", "5045"],
        "Utilities & Home Services": ["4900", "7210", "7349", "1711","5719","5712","3174","3144"],

        "Machinery & Tools": ["3058", "3066", "3075", "3504", "3509", "3596","5211","5261","3256"],
        
        "Rail & Bus Transport": ["3722", "3771", "3775", "4111", "4112", "4121", "4131", "4784", "3780"],
        
        "Telecommunications & Media": ["4814", "4829", "4899"],

        "Electronics & Appliances": ["3640","3684","5732","5251","5722"],
        "Automotive & Transportation Services": ["5533", "5541", "7531", "7538", "7542", "7549"],
        
        "Restaurants & Eating Places": ["5812", "5813", "5814"],
        
        "Clothing & Fashion": ["5621", "5651", "5655", "5661", "5977", "5932", "5947", "7230","3132"],
        
        "Movies & Theaters": ["7832", "7922", "5192", "5942"],
        "Sports & Recreational Activities": ["7801", "7802", "5941", "5970","7996","7995","5733"],
        
        "Medical & Healthcare Services": ["8011", "8021", "8041", "8043", "8049", "8062", "8099", "5912", "5921"],
        
        "Legal & Financial Services": ["8111", "8931", "7276", "7393","6300"],
        
        "Hotels & Accommodation": ["7011", "4411", "4511", "4722"],
        
        "Postal Services - Government Only": ["9402"],
    }

    # Create a reverse lookup: map each code to its category (subgroup)
    code_to_category = {}
    for category, codes in subgroup_mapping.items():
        for code in codes:
            code_to_category[code] = category

    # Prepare the CSV data
    # The CSV will have three columns: Code, Product, Category
    csv_rows = []
    header = ["Code", "Product", "Category"]
    csv_rows.append(header)

    for code, product in data.items():
        # Use the category from our mapping; if a code isn't found, assign "Uncategorized"
        category = code_to_category.get(code, "Uncategorized")
        csv_rows.append([code, product, category])

    # Write the CSV file
    # Uncomment and executing the code to write the CSV file
    # with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(csv_rows)

    # Read the CSV file into a DataFrame
    output = pd.read_csv("./data/processed/output.csv")

    # Merge the transactions DataFrame with the output DataFrame
    merged_trans = pd.merge(tran_df, output, left_on='mcc', right_on='Code', how='left')

    # Group by 'client_id' and 'Category', summing the 'amount', and resetting the index
    # This extracts the total spending per client and category
    category_spend = merged_trans.groupby(['client_id', 'Category'])['amount'].sum().reset_index()
    
    # Pivot the table to have 'client_id' as index and 'Category' as columns
    spending = category_spend.pivot(index='client_id', columns='Category', values='amount').fillna(0).reset_index()
    spending.columns.name = None

    # Merge the spending DataFrame with the final DataFrame, using a left join on 'client_id'
    # Not all custiomers will have spending data, so we use left join
    final = pd.merge(final_df, spending, left_on='id', right_on='client_id', how='left')

    # Remove the 'client_id' column from the final DataFrame
    final.drop(columns=['client_id','id'], inplace=True)
    # Uncomment and Export the final DataFrame to a CSV file
    final.to_csv("./data/processed/final_data.csv", index=False)

    print("Final dataset for recommendation done!\n")

if __name__ == "__main__":
    main()
