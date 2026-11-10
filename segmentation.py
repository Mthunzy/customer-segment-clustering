"""
CUSTOMER SEGMENTATION | UNSUPERVISED LEARNING
Author: Mthunzi Ntombela
GitHub: https://github.com/Mthunzy
Dataset: Mall Customers - 200 customers
Goal: Group customers into segments using K-Means Clustering
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
print("Loading data...")
url = "https://raw.githubusercontent.com/plotly/datasets/master/mall_customers.csv"
df = pd.read_csv(url)

print(f"Total customers: {len(df)}")
print(df.head())
print(df.info())

# ------------------------------------------------
# 2. EXPLORE DATA
# ------------------------------------------------
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# ------------------------------------------------
# 3. PREPARE FEATURES
# ------------------------------------------------
# We cluster based on Annual Income and Spending Score
# These are the 2 most important features for marketing
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print(f"\nFeatures used: {list(X.columns)}")

# Scale features so income and spending are on same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled!")

# ------------------------------------------------
# 4. FIND BEST NUMBER OF CLUSTERS - Elbow Method
# ------------------------------------------------
print("\nFinding best number of clusters (Elbow Method)...")
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    print(f"K={k}, Inertia={kmeans.inertia_:.2f}")

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method - Finding Best K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('elbow_curve.png')
print("Elbow curve saved as elbow_curve.png")

# Elbow is at K=5 - Best number of segments
print("\nBest K = 5 (elbow point)")

# ------------------------------------------------
# 5. TRAIN FINAL MODEL - K-Means with K=5
# ------------------------------------------------
print("\nTraining K-Means with K=5...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("Model trained!")

# ------------------------------------------------
# 6. ANALYSE CLUSTERS
# ------------------------------------------------
print("\n--- Customers per Cluster ---")
print(df['Cluster'].value_counts().sort_index())

print("\n--- Average Income & Spending per Cluster ---")
summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean().round(1)
print(summary)

# ------------------------------------------------
# 7. NAME THE SEGMENTS - Business Interpretation
# ------------------------------------------------
print("\n--- BUSINESS SEGMENTS ---")
print("Cluster 0: High Income, High Spending = PREMIUM / LUXURY LOVERS")
print("Cluster 1: High Income, Low Spending = CAREFUL RICH - Need incentives")
print("Cluster 2: Low Income, Low Spending = BUDGET SHOPPERS")
print("Cluster 3: Low Income, High Spending = IMPULSE BUYERS - Risky but loyal")
print("Cluster 4: Medium Income, Medium Spending = AVERAGE SHOPPERS")

# ------------------------------------------------
# 8. VISUALISE CLUSTERS
# ------------------------------------------------
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        s=50, label=f'Cluster {cluster}'
    )

plt.title('Customer Segmentation - 5 Customer Groups')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig('customer_segments.png')
print("\nSegmentation chart saved as customer_segments.png")

print("\n--- CONCLUSION ---")
print("Shops can now target each group differently:")
print("- Send luxury offers to Premium Lovers")
print("- Send discounts to Careful Rich to make them spend")
print("- Send budget deals to Budget Shoppers")
print("This increases sales without wasting marketing budget.")
