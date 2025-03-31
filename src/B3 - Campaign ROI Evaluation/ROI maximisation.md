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
