## 1. Overview of Campaign ROI maximising strategy
In this section, campaign ROI evaluation focuses on a customer-centric framework, where Return on Investment (ROI) = Customer Lifetime Value (CLV) – Customer Acquisition Cost (CAC). CLV represents the total revenue a customer is expected to generate over their relationship with the business, which is predicted by the BG/NBD model while CAC reflects the cost of acquiring that customer through marketing efforts. Thus, model can output the campaign ROI with input of campaign cost and customer responses in terms of purchases made after the campaign. 

By leveraging key customer behavior metrics such as purchase frequency, monetary value, recency, tenure and CLV, we can also segment customers to maximise campaign ROI. This allows for data-driven marketing decisions, thus optimizing campaign parameters like budget allocation, content, platforml and target audience. Ultimately, this approach ensures that marketing campaigns drive sustainable growth and profitability by maximizing ROI through personalized engagement and targeted retention strategies.

## 2. Customer Segementation
We decided to implement a K-means clustering to segment customers based on their behaviour. We used the Elbow Method and Silhouette score to identify the optimum number of clusters.

### 2.1 Identifying the optimal K clusters
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Select relevant features
X = grouped[['recency', 'frequency', 'monetary_value', 'T', 'predicted_1yr_clv']]

# Elbow Method: WCSS for different k values
wcss = []
silhouette_scores = []
K_range = range(2, 11)  # Testing k from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    wcss.append(kmeans.inertia_)  # WCSS
    silhouette_scores.append(silhouette_score(X, labels))  # Silhouette Score

# Plot Elbow Method
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method")

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', color='r')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")

plt.tight_layout()
plt.show()
```
![Kmeans](https://github.com/user-attachments/assets/f4df5795-c195-426f-a1f0-71e48156287c)
From the plot of elbow method, WCSS starts to flatten from K=5 onwards, where the rate of improvement slows, indicating an optimal balance between segmentation detail and model simplicity. 

Although the Silhoutte score shows a sharp drop for K=7, we feel having 7 clusters might lead to excessively fragmented groups, making it harder to explain customer behavior by looking at each group's statistical distributions, thus hindering creation of actionable marketing strategies. 

Therefore, we decided to segment customers into 5 clusters so that we can assign meaningful labels such as High Purchase, Loyal Customers, Potential Loyalists, Recently Converted, and At-Risk Customers based on their behaviour statistics.

### 2.2 Segmentation results
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = grouped[['recency', 'frequency', 'monetary_value', 'T', 'predicted_1yr_clv']] 
kmeans = KMeans(n_clusters=5, random_state=42)
grouped['cluster'] = kmeans.fit_predict(X)

grouped['cluster'].value_counts().plot(kind='bar')
plt.show()
```
![cluster number](https://github.com/user-attachments/assets/dd7eee00-222f-466f-bfed-7e3152ee43d5)

```python
# Visualisation of customer segments distribution
features = X[['recency', 'frequency', 'monetary_value', 'T', 'predicted_1yr_clv']]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.suptitle('Distributuion of Features by Cluster')
axes = axes.flatten()

for i, feature in enumerate(features):
    grouped.boxplot(column=feature, by='cluster', ax=axes[i])
    axes[i].set_title(f'{feature} by Cluster')
    axes[i].set_xlabel('Cluster')
    axes[i].set_ylabel(feature)

# Hide the unused subplot (the last one)
axes[-1].axis('off')

plt.show()
```
![Cluster features](https://github.com/user-attachments/assets/6098ee37-41bf-4119-abe3-afbe8f752281)

```python
# Inspection of statistics of each cluster
cluster_summary = grouped.groupby('cluster').agg({
    'T':'mean',
    'monetary_value': 'mean',
    'frequency': 'mean',
    'recency': 'mean',
    'predicted_1yr_clv': 'mean'
})
print(cluster_summary)
```
<img width="590" alt="Screenshot 2025-03-31 at 13 04 48" src="https://github.com/user-attachments/assets/b8e2dfe0-30db-4bba-800b-91a82a6f45d8" />

From the statistical and graphical distribution of customer segments, we can see their is a generally good distinction across segments, though each segment can have a bit of variability. Let us interpret the segmentation results below.

## Interpretation of segmentation results
