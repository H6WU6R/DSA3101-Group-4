# Campaign Impact Analysis: Architecture and Design Report

## 1. Pain Points in Traditional Campaign Assessment
Traditional campaign analysis often relies on discrete metrics—like raw conversion rates or isolated engagement figures—which can lead to a fragmented view of marketing effectiveness. Key limitations include:
- **Siloed Metrics**: Evaluations based on individual KPIs fail to capture the holistic customer journey.
- **Limited Actionability**: Disparate indicators make it challenging to derive unified, actionable insights that drive strategy.

Our composite score framework addresses these issues by aggregating multiple KPIs into a single metric, offering a comprehensive measure of campaign performance.

---

## 2. Updated Features for Logistic Regression and PCA

### 2.1 Logistic Regression Features

```python
features_for_lr = [
    'ClickThroughRate',
    'ConversionRate',
    'WebsiteVisits',
    'PagesPerVisit',
    'TimeOnSite',
    'SocialShares',
    'EmailOpens',
    'EmailClicks',
    'PreviousPurchases',
    'LoyaltyPoints'
]
```

These variables capture both immediate engagement (clicks, site activity, social shares) and historical indicators (previous purchases, loyalty points) to produce a robust conversion probability estimate.

### 2.2 PCA Columns for KPI Weighting

```python
kpi_columns = [
    'ClickThroughRate',
    'ConversionRate',
    'WebsiteVisits',
    'PagesPerVisit',
    'TimeOnSite',
    'SocialShares',
    'EmailOpens',
    'EmailClicks',
    'ConversionProbability'
]
```

After computing a Conversion Probability via logistic regression, we treat it as an additional KPI. This set of 9 features is standardised and fed into PCA to derive data-driven weights.

---

## 3. System Architecture Overview

1. **Data Ingestion & Preprocessing**  
   - Collects and cleans marketing data, including user engagement metrics (CTR, website visits, etc.) and historical behavior (purchases, loyalty points).

2. **Predictive Modeling Layer**  
   - Uses logistic regression on `features_for_lr` to predict a `ConversionProbability` for each customer.  
   - Model is trained and validated using standard techniques (train-test split, cross-validation).

3. **Normalization & PCA Layer**  
   - Normalizes the KPIs listed in `kpi_columns`.  
   - Applies PCA to extract the primary components and uses the loadings from PC1 to determine weights for each KPI.

4. **Composite Score Computation**  
   - Computes a weighted sum of the KPIs (including the newly computed `ConversionProbability`).  
   - Scales the composite score (e.g., 1–10) for intuitive interpretation.

5. **Feedback & Optimization**  
   - Monitors the composite score across different customer segments and campaigns.  
   - Iteratively refines model hyperparameters, PCA weighting, and derived metrics based on performance data.

---

## 4. Algorithm Design and Implementation

### 4.1 Conversion Probability Computation

1. **Logistic Regression**  
   - Train a logistic regression model using `features_for_lr`.  
   - Obtain `ConversionProbability = lr.predict_proba(X)[:, 1]`.

2. **Feature Importance**  
   - Analyze regression coefficients to identify which features most influence conversion.  
   - Optionally, refine or remove low-impact features.

### 4.2 PCA-Based Weighting

1. **Standardize KPI Data**  
   - Use `StandardScaler` on columns in `kpi_columns`.

2. **Run PCA**  
   - Extract the first principal component (PC1).  
   - Take absolute values of PC1 loadings, then normalize so they sum to 1.

3. **Compute Composite Score**  
   - Weighted sum of standardized KPIs: \(\sum_{i=1}^{n} w_i \times kpi_i\).  
   - scale to a 1–10 range.

---

## 5. Business Impact

- **Unified Metric**: Consolidates disparate KPIs into a single success score, providing a more holistic view of campaign effectiveness.
- **Adaptive Weighting**: PCA-driven weights adjust as underlying data shifts, ensuring continued relevance.
- **Actionable Insights**: Identifies top drivers of success, guiding targeted improvements in campaign design.
- **Strategic Resource Allocation**: Enables focused marketing efforts on the channels, messages, and audiences that yield the greatest ROI.
