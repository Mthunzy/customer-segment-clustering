# Customer Segmentation with K-Means Clustering

## Author
**Mthunzi Ntombela**  
Johannesburg, Gauteng | 083 981 6871 | m2nzhiey@gmail.com  
GitHub: https://github.com/Mthunzy  
LinkedIn: linkedin.com/in/mthunzi-ntombela-51646b145

---

## The Problem
A shopping mall has 200 customers. They all get the same marketing.

But a rich customer who loves spending is NOT the same as a budget shopper.

Can we group customers into segments so marketing can be targeted?

---

## The Dataset
- **Source:** Mall Customers Dataset
- **Size:** 200 customers
- **Features Used:** Annual Income (k$) + Spending Score (1-100)

---

## What I Did
1. Loaded and explored customer data with Pandas
2. Selected Income + Spending Score as clustering features
3. Scaled features with StandardScaler
4. Used Elbow Method to find best K = 5 clusters
5. Trained K-Means Clustering Model (K=5)
6. Analysed each cluster's average income, spending and age
7. Named each segment for business use
8. Visualised clusters with Matplotlib

---

## The 5 Customer Segments Found

| Cluster | Who They Are | Marketing Strategy |
| :--- | :--- | :--- |
| 0 | Premium / Luxury Lovers - High Income, High Spending | Send luxury offers, VIP invites |
| 1 | Careful Rich - High Income, Low Spending | Send discounts to unlock spending |
| 2 | Budget Shoppers - Low Income, Low Spending | Send budget deals, value packs |
| 3 | Impulse Buyers - Low Income, High Spending | Loyalty programme, careful credit offers |
| 4 | Average Shoppers - Medium Income, Medium Spending | General promotions |

---

## Tools Used
Python | Pandas | Scikit-Learn | K-Means Clustering | Matplotlib | Unsupervised Learning

---

## Project Structure
