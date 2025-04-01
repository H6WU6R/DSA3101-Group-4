# DSA3101-Group-4: Machine Learning for Personalized Marketing Campaigns in Banking

- [Project-3: Machine Learning for Personalized Marketing Campaigns in Banking](#project-3-machine-learning-for-personalized-marketing-campaigns-in-banking)
  - [1. Project Overview](#1-project-overview)
  - [2. Technical Implementation](#2-technical-implementation)
    - [2.1 Branch Usage Instruction](#21-branch-usage-instruction)
    - [2.2 Main Branch_Repository Structure](#22-main-branch-repository-structure)
    - [2.3 Setup Instruction](#23-setup-instruction)
  - [3. Deployment](#3-deployment)
    - [3.1 Web Application](#31-web-application)
    - [3.2 Docker Instructions](#32-docker-instructions)
  - [4. Data Understanding](#4-data-understanding)
    - [4.1 Data Preparation](#41-data-preparation)
    - [4.2 Data Dictionaries](#42-data-dictionaries)

## 1. Project Overview

Group 4, consisting of 10 data scientists, aims to address the challenge banks face in requiring robust, data-driven insights to enhance customer engagement, increase conversion rates, and improve marketing strategies. Traditional marketing practices are often inefficient, leading to poor resource allocation and missed revenue opportunities.  

To overcome these challenges, we are developing an AI-driven system that leverages advanced machine learning techniques to:  
- Effectively segment customers  
- Accurately predict future needs  
- Dynamically optimize marketing campaigns in real-time using detailed behavioral and preference data  

## 2. Technical Implementation

### 2.1 Branch Usage Instruction

This repository is organized to manage various problem codes and related tasks efficiently. Here's an overview of the relationship between the branches and how to use the repository:

1. **Main Branch**:
   - To run the entire program, navigate to the `main` branch and execute the script located at `source/main.py`.
   - To access code for specific problems, go to the respective folder under the `source` directory. For example, for problem **A1-Customer-Segmentation**, go to `source/A1-Customer-Segmentation`.
   - Raw and processed data can be found in the `source/data` directory.

3. **Individual Problem Branches**:
   - For detailed exploration of individual problems, including notebooks for Exploratory Data Analysis (EDA) and plots, checkout the specific problem branch. 
   - For example, to view details of **A1-Customer-Segmentation**, use the following command to switch to the relevant branch:
     ```bash
     git checkout A1-Customer-Segmentation
     ```

This structure ensures that the main program remains stable while individual branches allow for the development and testing of specific problems and tasks.

### 2.2 Main Branch Repository Structure

```plaintext
DSA3101-Group-Project  
│
├── src/
│   ├── A1-Customer-Segmentation/
│   │   └── A1_main.py
│   │
│   ├── A2-Customer-Engagement/
│   │   └── A2_main.py
│   │
│   ├── A3-Behavioral-Patterns/
│   │   ├── A3_main.py
│   │   ├── A3_trainmodel.py
│   │   └── A3_dataclean.py
│   │
│   ├── A4-Campaign-Impact-Analysis/
│   │   ├── A4_main.py
│   │   ├── A4_trainmodel.py
│   │   └── A4_dataclean.py
│   │
│   ├── A5-Segmentation-Updates/
│   │   ├── app.py                        # Main Dash application entry point
│   │   ├── assets/                       # Static assets for styling and images
│   │   │   ├── App Logo.webp             # Application logo
│   │   │   ├── Color.jpg                 # Background or color reference image
│   │   │   └── style.css                 # Custom CSS for styling the application
│   │   ├── pages/                        # Dashboard page components
│   │   │   ├── extract.py                # Data upload and processing interface
│   │   │   ├── home.py                   # Landing page with navigation
│   │   │   ├── individual_dashboard.py   # Individual customer analysis view
│   │   │   ├── nav.py                    # Navigation bar component
│   │   │   ├── overview_pages.py         # Additional overview page components
│   │   │   ├── overview.py               # Segmentation overview and analytics
│   │   │   └── topbar.py                 # Top bar UI component
│   │   └── src/                          # Core functionality
│   │       ├── __init__.py               # Package initialization
│   │       ├── prompts.py                # System prompts and templates
│   │       ├── recommendation.py         # Recommendation engine
│   │       └── segmentation.py           # Segmentation model and logic
│   │
│   ├── B1-Predicting-Customer-Preferences/
│   │   ├── B1_main.py
│   │   ├── B1_trainmodel.py
│   │   └── B1_dataclean.py
│   │
│   ├── B2-Campaign-Optimization/
│   │   └── B2.md
│   │
│   ├── B3-Measuring-Campaign-ROI/
│   │   ├── B3_main.py
│   │   ├── B3_trainmodel.py
│   │   └── B3_dataclean.py
│   │
│   ├── B4-Cost-Effectiveness-of-Campaigns/
│   │   ├── B4_main.py
│   │   ├── B4_trainmodel.py
│   │   └── B4_dataclean.py
│   │
│   ├── B5-Customer-Retention-Strategies/
│   │   ├── B5_main.py
│   │   ├── B5_trainmodel.py
│   │   └── B5_dataclean.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── Data1.csv
│   │   └── Data2.csv
│   │
│   └── processed/
│       └── (processed files)
│
├── docs/
│   └── README.md
│
├── database
│   ├── populate_db.py
│   ├── access_database.py
│   └── README.md
├── requirements.txt
├── Dockerfile
├── api.py
└── .gitignore
```
### 2.3 Setup Instruction

To set up the project on a local machine, follow the steps below:

1. **Ensure Python 3.13 is installed**.  
   If not, you can visit the [Python website](https://www.python.org/) for instructions on installation.  
   Once installed, you can verify your version of Python by running the following in your terminal:
   
   ```bash
   python --version
   ```
2. **Ensure Git is installed**
If you do not have Git installed, visit the [Git website](https://git-scm.com/) for instructions on installation. Once installed, you can verify your version of Git by running the following in your terminal:

```bash
git --version 
```

3. **Clone the respository** You may do this using SSH:

```bash
git clone git@github.com:H6WU6R/DSA3101-Group-4.git
```
Alternatively, you may clone using HTTPS:

```bash
git clone https://github.com/H6WU6R/DSA3101-Group-4.git
```

4. **Set working directory**

Set your working directory to the folder containing the cloned repository:

```bash
cd DSA3101-Group-4
```

5. **Create virtual environment**

```bash
python -m venv.
```

6. **Activate the virtual environment**

```bash
venv\Scripts\activate
```

7. **Install necessary packages**

```bash
pip install -r requirements.txt
```

8. **Run the main program**

```bash
python -m src.main
```

9. **Deactivate your virtual environment**

```bash
deactivate
```

## 3. Deployment
### 3.1 Web Application

We used Dash to create a web application, which consists of the following:

- **Interactive Visualizations**  
  All the important visualizations designed in segmentation are integrated into the web application, allowing users to explore customer segmentation, behavioral patterns, and campaign performance interactively.

- **Executive Dashboard**  
  An executive dashboard is provided to track the performance of customer segmentation and marketing campaigns, based on our chosen set of features.  

- **User Interface for Individual Customer Analysis**  
  An interface for users to analyze individual customer profiles, including personalized recommendations and behavioral insights.

- **Integration with Google AI API**  
  The web application leverages Google AI API to enhance its capabilities, including:
  - **Natural Language Processing (NLP)**: Used for analyzing customer profile and extracting insights.
  - **Prediction Models**: Used to improve segmentation accuracy and provide real-time recommendations.

This integration allows the application to deliver more accurate and intelligent insights, improving the overall user experience.

### 3.2 Docker Instructions

To build and run the necessary Docker containers, follow the steps below:

1. **Install Docker**

If you do not have Docker installed, visit the [Docker website](https://www.docker.com/get-started) for instructions on installation. Once installed, you can verify your version of Docker by running the following command in your terminal:

```bash
docker --version
```

2. **Set Working Directory**

Set your working directory to the folder containing the cloned repository:

```bash
cd DSA3101-Group-4
```

3. **Build Docker Image**

```bash
docker build -t dsa3101-group4 .
```

4. **Run Image in Container**

```
docker run --name DSA3101-Project-4 -w /app/src dsa3101-group4 python main.py
```

## 4. Data Understanding
### 4.1 Data Acquisition

To better understand and segment bank customer needs—thereby personalizing and optimizing marketing campaigns—we have identified and utilized four relevant datasets from Kaggle. 
- For customer segmentation, engagement, behavioral analysis and campaign cost-effectiveness evaluation, we used [digital_marketing_campaign_dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset).
- To predict customer preferences, we are using the [financial transactions dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets).
- For measuring campaign ROI, we rely on the [transactions_and_customer_insights](https://www.kaggle.com/datasets/rishikumarrajvansh/marketing-insights-for-e-commerce-company?select=CustomersData.xlsx).
- Finally, to support customer retention strategy development, we are leveraging the [bank customer churn data](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)。

### 4.2 Data Preparation and Feature Engineering

For each dataset, we conducted tailored feature engineering based on the specific requirements of the associated tasks.  This involves selecting, modifying, or creating new variables to better represent the underlying patterns in the data. Please refer to ?? for a detailed documentation of these processes.
Please refer to [??] for detailed documentation of the processing pipelines and transformations used.

#### 4.2.1 Digital Marketing Campaign Dataset
1. Data Loading: Imported dataset using `pd.read_csv()`.
2. Column Dropping: Removed `AdvertisingPlatform`, `AdvertisingTool`, and `CustomerID` to retain only relevant features.
3. Categorical Encoding: Applied one-hot encoding to `Gender`, `CampaignChannel`, and `CampaignType`.
4. Feature Scaling: Standardized all features using `StandardScaler` for model compatibility.
5. Feature Engineering:
  - `email_ctr`  
    - Formula: `EmailClicks` / `EmailOpens` 
    - Purpose: Measures the effectiveness of email campaigns by tracking how often recipients click on links. 
  - `engagement_depth`
    - Formula: `PagesPerVisit` * `TimeOnSite`  
    - Purpose: Quantifies how deeply a user interacts with the site, reflecting both the number of pages viewed and the total time spent.
  - `social_propensity` 
    - Formula: `SocialShares` / `WebsiteVisits`  
    - Purpose: Captures the customer’s tendency to share or repost content, indicating the potential for viral or word-of-mouth engagement.


#### 4.2.2 Financial Transactions Dataset
1. Carry out data cleaning on `cards_data.csv`, convert `credit_limit` to numeric value by removing \$ in the beginning. Unnecessary columns like `card_number`, `cvv` and etc have been dropped to reduce dimensionality. Data aggregation has been done to show each customer's total credit limit, debit and prepaid savings. Resultant dataset saved temporarily as `cards_new`.
2. Carry out data cleaning on `transactions_part1.csv` and `transactions_part2.csv`, convert `amount` to numeric value by removing \$ in the beginning, convert `date` to datetime data types. Erroneous transactions are removed.
3. Carry out data cleaning on `users_data.csv`, change gender to binary representation, drop unnecessary columns like `address` and `per_capita_income`. Columns like `per_capita_income`, `yearly_income`, `total_debt` are converted to numeric value by removing \$ in the beginning. Merge with `cards_new`, using correlation matrix to remove columns that are positively (i.e. `id` and `client_id`) or negatively (i.e. `current_age` and `birth_year`) related. Resultant dataset saved temporarily as `final_cards`.
4. All 3 datasets are merged and saved as `final_data.csv`.
5. Manual grouping of mcc codes have been done and each mcc is assigned with a category, the data is saved as `output.csv`. This dataset is then joined with `final_data.csv`, to illustrate how much a user spend in each category. KNNImputer is used to compute the missing values of the spending use demographic data. Thereafter, we had come out with our own set of label construction rules, constructing ordinal label for each financial product that we have derived from the datasets. Resultant dataset saved as `imputed_data_with_label.csv`.
#### 4.2.3 Transactions and Customer Insights Dataset
  1. Carry out data cleaning on `Online_Transactions.csv` to ensure unique rows of transactions and appropriate data type. Only relevant columns are kept. Join on `CustomersData.xlsx` by CustomerID, the resultant dataset is saved as `clv_prediction.csv`
  2. Carry out feature engineering by computing essential features for predicting customer lifetime value, including Recency, Frequency, Monetary_value, T.
     
#### 4.2.4 Bank Customer Churn Dataset
1. Dropped columns that are not relevant, have privacy issues, or are perfectly correlated with our label `Exited`: `RowNumber`, `CustomerId`,`Surname`, and `Complain`.
2. Created a new feature called `Income_bin` from `EstimatedSalary`, which assign customers to their corresponding income interval. Then, merged with `digital_marketing_campaign_dataset.csv` after segmentation on `Age`, `Gender`, and `Income_bin`.
3. Applied Ordinal Ecoding to transform categorical features: `Gender`,`Geography`, `Card Type`.
4. Used SMOTE to synthesize a more balanced traning data
   
### 4.3 Data Dictionaries
This section contains a list of processed datasets for each of the CSV files stated in Section 4.2, in order of appearance.
   1. `digital_marketing_campaign_dataset.csv`
  
| Field Name | Description | Data Type | Allowed Values | Example |
  |:---|:---|:---:|:---|:---:|
| `CustomerID` | Unique identifier for each customer | int64 | Positive integers | 8000 |
| `Age` | Customer age in years | int64 | 18-100 | 56 |
|` Gender`| Customer gender | object | ['Female', 'Male', 'Other'] | 'Female' |
| `Income`| Annual income in local currency | float64 | Positive numbers | 136912.0 |
| `CampaignChannel` | Marketing channel used | object | ['Social Media', 'Email', 'PPC', 'Referral', 'SEO'] | 'Social Media' |
| `CampaignType` | Campaign objective type | object | ['Awareness', 'Retention', 'Conversion'] | 'Awareness' |
| `AdSpend` | Amount spent on advertising for this customer (currency) | float64 | Positive numbers | 6497.87 |
| `ClickThroughRate` | Ratio of ad clicks to impressions | float64 | 0.0-1.0 | 0.0439 |
| `ConversionRate` | Ratio of conversions to clicks | float64 | 0.0-1.0 | 0.0880 |
| `WebsiteVisits` | Number of website visits during campaign | int64 | Non-negative integers | 0 |
| `PagesPerVisit` | Average pages viewed per visit | float64 | Positive numbers | 2.399 |
| `TimeOnSite` | Average time spent on site (minutes) | float64 | Positive numbers | 7.396 |
| `SocialShares` | Number of social media shares | int64 | Non-negative integers | 19 |
| `EmailOpens`| Number of marketing emails opened | int64 | Non-negative integers | 6 |
| `EmailClicks`| Number of clicked links in emails | int64 | Non-negative integers | 9 |
| `PreviousPurchases` | Count of purchases before campaign | int64 | Non-negative integers | 4 |
| `LoyaltyPoints` | Accumulated loyalty program points | int64 | Non-negative integers | 688 |
| `AdvertisingPlatform` | Platform used for ads | object | ['IsConfid', 'OtherPlatforms'] | 'IsConfid' |
| `AdvertisingTool` | Tool used for ad management | object | ['ToolConfid', 'OtherTools'] | 'ToolConfid' |
| `Conversion` | Whether conversion occurred (1/0) | int64 | [0, 1] | 1 |

  
  2.`cards_data.csv` 
  | Field Name | Description | Data Type | Allowed Values | Example |
  |:---|:---|:---|:---|:---:|
  | `id`|Unique identifier for the card record| int64| Positive integer| 1001|
| `client_id`| Identifier linking the card to a customer| int64| Positive integer| 17490|
| `card_brand`       | Brand of the card (e.g., Visa, MasterCard)                         | object    | Visa, MasterCard, Amex, Discover| Visa |
| `card_type`        | Type of card issued (credit, debit, prepaid)                       | object    | Credit, Debit, Debit (Prepaid)| Credit             |
| `card_number`      | card number for each card| int64    | Positive integer              | 4336733185475861 |
| `expires`          | Expiration date of the card                                        | object    | Valid date in MM/YYYY format  | 02/2020     |
| `cvv`              | Card Verification Value (security code)                           | int64     | 3-digit number| 123                  |
| `has_chip`         | Indicates whether the card is equipped with a chip             | object      | YES, NO| YES |
| `num_cards_issued` | Total number of cards issued to the customer                        | int64     | Non-negative integer | 2   |
| `credit_limit`     | Maximum credit available on the card for credit card, deposit amount for debit and prepaid card      | float64   | Positive number                      | 5000.00              |
| `acct_open_date`   | Date the card account was opened                                   | object    | Valid date in MM/YYYY format    | 04/2014  |
| `year_pin_last_changed` | Year when the card PIN was last updated                        | int64     | Positive integer| 2008 |
| `card_on_dark_web` | Indicator if the card details were found on the dark web             | object    | YES, NO | YES|


3.`transactions_part1.csv` and `transactions_part2.csv`
  | Field Name | Description | Data Type | Allowed Values | Example |
  |:---|:---|:---|:---|:---:|
  | `id`           | Unique identifier for the transaction                              | int64     | Positive integer                         | 7475327               |
| `date`         | Timestamp when the transaction occurred                             | object    | Valid date/time format | "2010-01-01 00:01:00"  |
| `client_id`    | Identifier of the customer who made the transactions| int64     | Positive integer| 561|
| `card_id`      | Identifier for the card used in the transaction                        | int64     | Positive integer| 2972|
| `amount`       | Transaction amount in US dollar                                 | object   | Positive and negative numbers with dollar sign | $200.00            |
| `use_chip`     | Indicates if the transaction used chip authentication                  | object      | Swipe Transaction, Online Transaction, Chip Transaction | Swipe Transaction|
| `merchant_id`  | Unique identifier for the merchant involved                           | int64     | Positive integer                         | 61195|
| `merchant_city`| City where the merchant is located or "ONLINE" | object    | String city names| Canton|
| `merchant_state`| State or region of the merchant | object    | String state abbreviation name | NY|
| `zip`| Merchant’s ZIP or postal code| float64    | Valid ZIP code format                    | 47805.0 |
| `mcc` | Merchant Category Code representing the business type                  | int64     | Standard MCC (4 digits) | 5411                  |
| `errors`       | Error codes or messages related to the transaction (if applicable)       | object    | Strings indicating the error status or null| Insufficient Balance |


4.`users_data.csv`
  | Field Name | Description | Data Type | Allowed Values | Example |
  |:---|:---|:---|:---|:---:|
  | `id`             | Unique identifier for the user                                      | int64     | Positive integer | 825                |
| `current_age`    | Age of the customer at the time of data collection                    | int64     | Positive integer                          | 53                  |
| `retirement_age` | Expected or actual retirement age of the customer                      | int64     | Positive integer | 67                 |
| `birth_year`     | Year when the customer was born | int64     | Four-digit year                           | 1966                 |
| `birth_month`    | Month when the customer was born (numeric or abbreviated)               | int64    | Positive integer        | 7                    |
| `gender`         | Gender of the customer                                               | object    | Male, Female        | Female             |
| `address`        | Residential address of the customer                                  | object    | String of addresses| "9620 Valley Stream Drive" |
| `latitude`       | Latitude coordinate of the customer's address                        | float64   | -90 to 90                                  | 41.55            |
| `longitude`      | Longitude coordinate of the customer's address                       | float64   | -180 to 180                                | -122.64|
| `per_capita_income`| Per capita income of the customer                                   | object   | Positive number with dollar sign | $26790|
| `yearly_income`  | Annual income of the customer                                         | object   | Positive number with dollar sign| $59696|
| `total_debt`     | Total debt owed by the customer                                       | object   | Non-negative number Positive number with dollar sign| $222735|
| `credit_score`   | Credit score of the customer                                          | int64     | Positive integer| 772 |
| `num_credit_cards`| Number of credit cards owned by the customer | int64     | Non-negative integer | 3 |

  5. `Online_Sales.csv`

| Field Name         | Description                                              | Data Type | Allowed Values           | Example |
|:------------------|:---------------------------------------------------------|:---------|:------------------------|:--------:|
| `CustomerID`        | Unique identifier for the customer                      | int64    | Positive integer        | 17850    |
| `Transaction_ID`    | Unique identifier for each transaction                   | int64    | Positive integer        | 16679    |
| `Transaction_Date`  | Date when the transaction occurred                       | object   | Date in MM/DD/YYYY format | 1/1/2019 |
| `Product_SKU`       | Stock Keeping Unit (SKU) of the product                  | object   | Alphanumeric string     | GGOENEBJ079499 |
| `Product_Description` | Description of the product                            | object   | String                  | Nest Learning Thermostat 3rd Gen-USA - Stainless Steel |
| `Product_Category`  | Category of the product                                 | object   | String                  | Nest-USA |
| `Quantity`         | Number of units purchased                               | int64    | Positive integer        | 1        |
| `Avg_Price`        | Average price per unit of the product                   | float64  | Positive decimal        | 153.71   |
| `Delivery_Charges` | Shipping cost for the transaction                       | float64  | Positive decimal        | 6.5      |
| `Coupon_Status`   | Status of coupon usage for the transaction               | object   | Used, Not Used | Used     |  
  
  6. `CustomersData.xlsx`

| Field Name     | Description                                | Data Type | Allowed Values                  | Example |
|:--------------|:-----------------------------------------|:---------|:------------------------------|:--------:|
| `CustomerID`    | Unique identifier for the customer       | int64    | Positive integer               | 17850    |
| `Gender`        | Gender of the customer                   | object   | M, F                           | M        |
| `Location`      | Geographic location of the customer      | object   | City or State names            | Chicago  |
| `Tenure_Months` | Duration of customer relationship (months) | int64    | Positive integer               | 12       |

  7. `Customer-Churn-Records.csv`

| Field Name          | Description                                              | Data Type | Allowed Values               | Example          |
|:-------------------|:------------------------------------------------------|:----------|:----------------------------|:----------------:|
| RowNumber          | Unique identifier for the row                            | int64     | Positive integer            | 1                |
| CustomerId         | Unique identifier for the customer                       | int64     | Positive integer            | 15634602         |
| Surname            | Last name of the customer                                | object    | String                      | Hargrave         |
| CreditScore        | Credit score of the customer                             | int64     | 300-850                     | 619              |
| Geography          | Country of residence                                     | object    | France, Spain, Germany      | France           |
| Gender             | Gender of the customer                                   | object    | Male, Female                | Female           |
| Age                | Age of the customer                                      | int64     | Positive integer            | 42               |
| Tenure             | Number of years as a customer                            | int64     | 0-10                        | 2                |
| Balance            | Account balance                                          | float64   | Positive decimal or 0       | 0.0              |
| NumOfProducts      | Number of bank products used                             | int64     | 1-4                         | 1                |
| HasCrCard          | Indicates if the customer has a credit card              | int64     | 0 (No), 1 (Yes)             | 1                |
| IsActiveMember     | Indicates if the customer is an active member            | int64     | 0 (No), 1 (Yes)             | 1                |
| EstimatedSalary    | Estimated salary of the customer                         | float64   | Positive decimal            | 101348.88        |
| Exited             | Indicates if the customer churned                        | int64     | 0 (No), 1 (Yes)             | 1                |
| Complain           | Indicates if the customer has lodged a complaint         | int64     | 0 (No), 1 (Yes)             | 1                |
| Satisfaction Score | Customer satisfaction score (1-5)                        | int64     | 1, 2, 3, 4, 5               | 2                |
| Card Type          | Type of credit card held                                 | object    | DIAMOND, GOLD, SILVER, PLATINUM | DIAMOND       |
| Point Earned       | Reward points earned by the customer                     | int64     | Positive integer            | 464              |

  8. `mcc_code.json`
     - A dictionary with its key being the merchant industry code, the value being the name of the corresponding industry.

