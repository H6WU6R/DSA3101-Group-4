# Campaign Impact Analysis

## 1. Introduction

This report presents an analysis of digital marketing campaign data to identify key performance indicators (KPIs) that differentiate converters from non-converters. Our goal is to establish a framework for assessing customer engagement and conversion likelihood using both overall and cluster-specific metrics.

## 2. Exploratory Data Analysis (EDA)

### 2.1 Data Overview
The dataset contains demographic, financial, and digital engagement variables:
- **Demographics:** Age, Gender, Income
- **Engagement Metrics:** ClickThroughRate, ConversionRate, WebsiteVisits, PagesPerVisit, TimeOnSite, SocialShares, EmailOpens, EmailClicks

### 2.2 Graphical EDA
We visualized the distribution of these variables using boxplots and count plots:
- **Boxplots:** Showed that converters generally exhibit higher median values for engagement metrics such as ClickThroughRate, ConversionRate, WebsiteVisits, PagesPerVisit, TimeOnSite, EmailOpens, and EmailClicks.
- **Count Plots:** For the Gender variable, the distribution revealed a higher count of female customers, though conversion rates appeared similar across genders.

*Example Graph:*
![EDA Graph](A4_Campaign_Impact_Analysis/visualisations/EDA.png)

### 2.3 Hypothesis Testing
We performed two-sample t-tests and Mann-Whitney U tests on numeric variables, and a chi-square test on the categorical variable (Gender):

- **Age & Income:**  
  - p-values: ~0.88 (Age), ~0.21 (Income)  
  - **Conclusion:** No significant difference between converters and non-converters.

- **Engagement Metrics:**  
  - p-values for ClickThroughRate, ConversionRate, WebsiteVisits, PagesPerVisit, TimeOnSite, EmailOpens, and EmailClicks were extremely low (p < 0.001).  
  - **Conclusion:** Converters exhibit significantly higher digital engagement.

- **SocialShares:**  
  - p-value ~0.31  
  - **Conclusion:** Not a strong predictor of conversion.

- **Gender:**  
  - Chi-square test p-value ~0.951  
  - **Conclusion:** No significant association between Gender and conversion.

## 3. KPI Threshold Selection

### 3.1 Methods Used
- **Youden's J Statistic:**  
  For each KPI, we computed the ROC curve on a training set and selected the threshold that maximized Youden's J (i.e., sensitivity – (1 – specificity)). This method provides an optimal cutoff value that best discriminates converters from non-converters.

- **Logistic Regression for Engagement Score:**  
  A logistic regression model was trained on the key engagement metrics. The model's coefficients were used to compute an engagement score (log-odds) for each customer, which was then scaled to a range of 0 to 10. This score represents a continuous measure of a customer's likelihood to convert.

### 3.2 Overall Thresholds (Population Level)
Using the overall data, the following fixed thresholds were established:
- **ClickThroughRate:** 0.097254  
- **ConversionRate:** 0.049689  
- **WebsiteVisits:** 11.0  
- **PagesPerVisit:** 2.981618  
- **TimeOnSite:** 4.989053  
- **EmailOpens:** 6.0  
- **EmailClicks:** 3.0  

## 4. Engagement Scoring & Evaluation

### 4.1 Engagement Score Computation
The logistic regression model (trained on the KPI columns) yielded a continuous engagement score for each customer. After scaling, customers received a **Scaled Engagement Score** (0–10). This score correlated strongly with the overall KPI hit count, indicating that higher digital engagement is closely linked with conversion.

### 4.2 KPI Hit Count Calculation
- **Overall KPI Hit Count:**  
  For each customer, we calculated the number of KPIs (based on the fixed overall thresholds) that were met. This was then used to compute the **Overall KPI Proportion**.

- **Cluster-Specific KPI Hit Count:**  
  The dataset was segmented into clusters (via prior segmentation). For each cluster, cluster-specific thresholds were determined (using Youden's J on a per-cluster basis). We then computed the number of KPIs met by each customer in their respective cluster.

### 4.3 Results Summary
- **Overall Analysis:**  
  Converters, on average, met a higher proportion of overall KPIs compared to non-converters. For example, the mean Overall KPI Proportion was ~0.76 for converters versus ~0.57 for non-converters (t-test, p < 0.001).
  *Example Graph:*
![KPI Hit Count Comparison](A4_Campaign_Impact_Analysis/visualisations/Overall%20KPI%20Hit%20Count%20vs%20Conversion.png)

- **Cluster-Specific Analysis:**  
  The cluster-specific thresholds were more variable. In some clusters, the thresholds were not meaningful due to small sample sizes or homogeneous behavior, leading to sparse threshold data.
![Cluster-Specific KPI Hit Proportion Comparison](A4_Campaign_Impact_Analysis/visualisations/Cluster%20KPI%20Hit%20Proportion%20vs%20Conversion.png)

## 5. Business Insights & Recommendations

### 5.1 Key Findings
- **Engagement Metrics as Conversion Drivers:**  
  Metrics like ClickThroughRate, ConversionRate, WebsiteVisits, PagesPerVisit, TimeOnSite, EmailOpens, and EmailClicks are significantly higher among converters, indicating that deeper and more frequent engagement drives conversion.

- **Demographic Factors:**  
  Age, Income, and Gender do not significantly differentiate converters from non-converters.

- **Cluster Variability:**  
  When segmented, certain clusters show different optimal thresholds for digital engagement. For example, high-value customers in one cluster may convert with only moderate engagement, whereas another segment might require significantly higher website dwell times.

### 5.2 Actionable Strategies
- **Segment-Specific Engagement Goals:**  
  Tailor marketing campaigns based on cluster insights. For clusters with low overall engagement, focus on boosting digital interactions. For clusters where high engagement is already observed, refine content for deeper exploration.

- **Revise Email Tactics:**  
  In clusters where email engagement metrics (opens and clicks) do not discriminate well (e.g., clusters with infinite thresholds), consider testing alternative formats or channels.

- **Enhance Website Experience:**  
  For clusters where metrics like WebsiteVisits and PagesPerVisit are significant, optimize site navigation and content to encourage deeper browsing and longer time on site.

- **Continuous Monitoring:**  
  Regularly update the overall and cluster-specific thresholds as customer behavior evolves to maintain a dynamic and responsive marketing strategy.

## 6. Visualizations
- **Optimal KPI Thresholds by Cluster:**  
  A bar plot was created to visualise the optimal KPI thresholds across clusters, highlighting where thresholds were computed and where data was sparse.
  ![Cluster-Specific Thresholds](A4_Campaign_Impact_Analysis/visualisations/Optimal%20KPI%20Thresholds%20by%20Cluster.png)

- **KPI Hit Count vs. Engagement Score:**  
  A scatter plot compared the overall KPI hit count to the scaled engagement score, demonstrating a strong correlation between digital engagement and conversion likelihood.
  ![KPI Hit Count vs. Engagement Score](A4_Campaign_Impact_Analysis/visualisations/KPI%20Hit%20Count%20vs%20Engagement%20Score.png)

---

## Conclusion

The analysis demonstrates that digital engagement metrics are robust indicators of conversion. While overall KPI thresholds offer a strong general framework, cluster-specific thresholds reveal nuances in customer behavior. These insights enable the design of targeted, segment-specific marketing strategies that can optimize campaign performance and drive higher conversion rates.

---
