# E-commerce Customer Segmentation Project

## Overview
This project analyzes an e-commerce dataset to segment customers based on their purchasing behaviors and patterns. Using various clustering techniques and machine learning algorithms, the analysis identifies distinct customer groups to enable targeted marketing strategies and personalized recommendations.

## Dataset
The dataset contains e-commerce transactions with the following fields:
- **InvoiceNo**: Unique invoice number for each transaction
- **StockCode**: Product code
- **Description**: Product description
- **Quantity**: Number of items purchased
- **InvoiceDate**: Date and time of purchase
- **UnitPrice**: Unit price of product
- **CustomerID**: Unique customer identifier
- **Country**: Country where the customer is located

## Features of the Analysis

### Data Cleaning and Preprocessing
- Handling missing values in CustomerID and Description fields
- Removing duplicate entries
- Identifying and managing cancelled transactions
- Handling anomalous stock codes
- Standardizing product descriptions
- Filtering out zero or negative unit prices

### Feature Engineering
- Recency: Days since last purchase
- Frequency: Total number of transactions
- Monetary Value: Total spend and average transaction value
- Product Diversity: Number of unique products purchased
- Shopping Patterns: Favorite shopping days and hours
- Geographic Segmentation: UK vs. non-UK customers
- Cancellation Behavior: Frequency and rate of cancellations
- Seasonal Spending: Monthly spending patterns and trends

### Outlier Detection
- Isolation Forest algorithm used to identify and remove outliers
- 5% of customers identified as outliers and excluded from clustering

### Dimensionality Reduction
- Principal Component Analysis (PCA) applied
- 6 principal components selected, capturing significant variance in the data

### Customer Segmentation
- K-means clustering algorithm applied to PCA-transformed data
- Optimal number of clusters (k=3) determined using:
  - Elbow method
  - Silhouette analysis
  - Calinski-Harabasz and Davies-Bouldin scores

### Cluster Analysis
- Radar charts to visualize cluster centroids
- Histogram distribution of features across clusters
- 3D visualization of customer clusters

### Recommendation System
- Product recommendations based on cluster-specific popular items
- Personalized recommendations for each customer based on prior purchase history

## Results

The analysis identified three distinct customer segments:

1. **High-Value Customers**: Frequent shoppers with high monetary value and product diversity
2. **Medium-Value Customers**: Regular shoppers with moderate spending
3. **Low-Value Customers**: Infrequent shoppers with lower monetary value

Each segment demonstrates unique shopping behaviors and preferences, enabling targeted marketing strategies.

## Files
- `customer_segmentation.py`: Main Python script containing the entire analysis
- `ecommerce_clustered.csv`: Output file with customer segmentation results
- `data-2.csv`: Original dataset (sample shown in README)

## Requirements
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - plotly
  - yellowbrick
  - scipy
  - tabulate

## Usage
1. Ensure all required libraries are installed
2. Place the dataset file in the appropriate directory
3. Run the script to perform analysis and generate segmented customer data

## Future Work
- Implement more advanced clustering algorithms (e.g., DBSCAN, Hierarchical)
- Develop a more sophisticated recommendation system using collaborative filtering
- Create a dashboard for interactive exploration of customer segments
- Conduct time-based analysis to track segment evolution