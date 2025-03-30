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
│   │   ├── A1_main.py
│   │   ├── A1_trainmodel.py
│   │   └── A1_dataclean.py
│   │
│   ├── A2-Customer-Engagement/
│   │   ├── A2_main.py
│   │   ├── A2_trainmodel.py
│   │   └── A2_dataclean.py
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
│   │   ├── A5_main.py
│   │   ├── A5_trainmodel.py
│   │   └── A5_dataclean.py
│   │
│   ├── B1-Predicting-Customer-Preferences/
│   │   ├── B1_main.py
│   │   ├── B1_trainmodel.py
│   │   └── B1_dataclean.py
│   │
│   ├── B2-Campaign-Optimization/
│   │   ├── B2_main.py
│   │   ├── B2_trainmodel.py
│   │   └── B2_dataclean.py
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
│   └── readme.md
│
├── database.py
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
docker build -t HAVE NOT ADD HERE !!!!
```

4.**Run Image in Container**

```
docker run --name HAVE NOT ADD HERE !!!!
```

## 4. Data Understanding
### 4.1 Data Acquisition

Based on our tasks of understanding and segmenting bank customer needs, thus personalizing and optimising bank marketing campaigns, we have researched and identified 4 relevant datasets from Kaggle. For customer segmentation, engagement, behavioral analysis and campaign cost-effectiveness evaluation, we used [digital_marketing_campaign_dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset); for predicting customer preferenes we are using [financial transactions dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets); for measuring campaign ROI we are using [transactions_and_customer_insights](https://www.kaggle.com/datasets/rishikumarrajvansh/marketing-insights-for-e-commerce-company?select=CustomersData.xlsx); for devising customer retention strategies we are using [bank customer churn data](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

### 4.2 Data Preparation and Feature Engineering

For each dataset, we have conducted feature engineering based on the requirements of the tasks. This involves selecting, modifying, or creating new variables to better represent the underlying patterns in the data. Please refer to ?? for a detailed documentation of these processes.

#### 4.2.1 Digital Marketing Campaign Dataset

#### 4.2.2 Financial Transactions Dataset

#### 4.2.3 Transactions and Customer Insights Dataset
  1. Carry out data cleaning on Online_Transactions.csv to ensure unique rows of transactions and appropriate data type. Only relevant columns are kept. Join on CustomersData.xlsx by CustomerID, the resultant dataset is saved as clv_prediction.csv
  2. Carry out feature engineering by computing essential features for predicting customer lifetime value, including Recency, Frequency, Monetary_value, T.
     
#### 4.2.4 Bank Customer Churn Dataset

### 4.3 Data Dictionaries
This section contains a list of processed datasets for each of the CSV files stated in Section 4.2, in order of appearance.
  1. 
  | Field Name | Description | Data Type | Allowed Values | Example |
  |:---------:|:----------:|:---------:|:-----------:|:-------:|
  |           |           |           |             |          |
  
  2.
  | Field Name | Description | Data Type | Allowed Values | Example |
  |:---------:|:----------:|:---------:|:-----------:|:-------:|
  |            |           |           |             |          |

  4. `clv_predictions.csv`

  | Field Name | Description | Data Type | Allowed Values | Example |
  |:---:|:---:|:---:|:---:|:---:|
  |CustomerID|Identifier for each customer|int64|Positive integer|12490|
  
  
  6. 
  | Field Name | Description | Data Type | Allowed Values | Example |
  |:---------:|:----------:|:---------:|:-----------:|:-------:|
  |            |           |           |             |          |
