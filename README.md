# DSA3101-Group-4: Machine Learning for Personalized Marketing Campaigns in Banking

- [Project-3: Machine Learning for Personalized Marketing Campaigns in Banking](#project-3-machine-learning-for-personalized-marketing-campaigns-in-banking)
  - [1. Project Overview](#1-project-overview)
  - [2. Technical Implementation](#2-technical-implementation)
    - [2.1 System Architecture](#21-system-architecture)
    - [2.2 Setup Instruction](#22-model-development)
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

### 2.1 System Architecture

DSA3101-Group-Project  

```plaintext
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
### 2.2 Setup Instruction

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

3. ** Clone the respository. You may do this using SSH:

```bash
git clone
```


## 4. Data Understanding
### 4.1 Data Acquisition

Based on our tasks of understanding and segmenting bank customer needs, thus personalizing and optimising bank marketing campaigns, we have researched and identified 4 relevant datasets from Kaggle. For customer segmentation, engagement, behavioral analysis and campaign cost-effectiveness evaluation, we used [digital_marketing_campaign_dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset); for predicting customer preferenes we are using [financial transactions dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets); for measuring campaign ROI we are using [transactions_and_customer_insights](https://www.kaggle.com/datasets/rishikumarrajvansh/marketing-insights-for-e-commerce-company?select=CustomersData.xlsx); for devising customer retention strategies we are using [bank customer churn data](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
