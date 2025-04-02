from sqlalchemy import create_engine
import pandas as pd
import json

############### Replace with your own credentials ###############
# db_credential='postgresql://username:password@localhost:5432' #
db_credential='postgresql://leizhenyu:0128@localhost:5432'
#################################################################

############ create same database name in PostgreSQL ############
db_name = '/DSA3101Bank'
#################################################################

data_path = './data/raw/'

def main(db_name, db_credential):
    
    ####################### A1 Dataset ##########################
    # db_name = '/DSA3101BankA1'

    print("A1 Data Loading...")
    df = pd.read_csv(data_path+'digital_marketing_campaign_dataset.csv')
    # Connect to database
    engine = create_engine(db_credential + db_name)
    # Load data to database
    table_name = 'a1_digital_marketing_campaign'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print("A1 Data loaded successfully!")

    ####################### B1 Dataset ##########################
    # db_name = '/DSA3101BankB1'

    print("B1 Data Loading...")

    print("  cards_data loading...")
    df = pd.read_csv(data_path+'cards_data.csv')
    engine = create_engine(db_credential + db_name)
    table_name = 'b1_cards_data'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print("  mcc_codes loading...")
    with open(data_path+'mcc_codes.json') as f:
        data = json.load(f)
    df = pd.DataFrame(list(data.items()), columns=['code', 'description'])
    engine = create_engine(db_credential + db_name)
    table_name = 'b1_mcc_codes'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print("  transactions_data loading (this data is huge)...")
    transactions_1 = pd.read_csv(data_path+'transactions_part1.csv')
    transactions_2 = pd.read_csv(data_path+'transactions_part2.csv')
    df = pd.concat([transactions_1, transactions_2], ignore_index=True)
    engine = create_engine(db_credential + db_name)
    table_name = 'b1_transactions_data'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print("  users_data loading...")
    df = pd.read_csv(data_path+'users_data.csv')
    engine = create_engine(db_credential + db_name)
    table_name = 'b1_users_data'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print("B1 Data loaded successfully!")

    ####################### B3 Dataset ##########################
    # db_name = '/DSA3101BankB3'

    print("B3 Data Loading...")
    # Connect to database
    engine = create_engine(db_credential + db_name)

    df = pd.read_csv(data_path+'Online_Sales.csv')
    table_name = 'b3_online_sales'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    df = pd.read_excel(data_path+'CustomersData.xlsx', header=0)
    table_name = 'b3_customers_data'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print("B3 Data loaded successfully!")

    ####################### B5 Dataset ##########################
    # db_name = '/DSA3101BankB5'

    print("B5 Data Loading...")
    df = pd.read_csv(data_path+'Customer-Churn-Records.csv')

    # Connect to database
    engine = create_engine(db_credential + db_name)

    # Load data to database
    table_name = 'b5_customer_churn_records'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print("B5 Data loaded successfully!")

    print("All Data loaded successfully!")

if __name__ == "__main__":
    main(db_name, db_credential)
