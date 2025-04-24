# -*- coding: utf-8 -*-

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from tabulate import tabulate
from collections import Counter


# %matplotlib inline

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

df = pd.read_csv('content/data-2.csv' , encoding="ISO-8859-1")

"""**Initial Data Analysis**"""

df.head(10)

df.info()

df.describe(include='all') # if we describe all many values are missing as can't be calculated

df.describe() # first describing numeerical values only

df.describe(include='object').T # here we are calculating categorical values

"""**Data Cleaning and Transformation**"""

missing_data = df.isnull().sum() # counting missing values
missing_percentage = (missing_data[missing_data > 0] / df.shape[0]) * 100 # converting count to percentage

missing_percentage.sort_values(ascending=True) # arranging in ascending order

fig, ax = plt.subplots(figsize=(15, 4)) # Dimensions for graph
ax.barh(missing_percentage.index, missing_percentage, color='#ff6200') # taking horizontal barchart with orange color

for value, name in zip(missing_percentage, missing_percentage.index): # labelling values on the graph
    ax.text(value,name , f"{value:.2f}%", fontweight='bold', fontsize=18, color='black')



plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)   # Giving title, label and displaying the graph
plt.xlabel('Percentages (%)', fontsize=16)
plt.show()

"""Handling NULL values

"""

df[df['CustomerID'].isnull() | df['Description'].isnull()] # finding null values in customerid and description

df = df.dropna(subset=['CustomerID', 'Description']) #dropping null values

df.isnull().sum().sum() #rechecking for null values

"""Handling Duplicate entries"""

duplicated_rows= df[df.duplicated(keep=False)] # Getting rows with duplicate entries
duplicated_rows

duplicated_rows_sorted= duplicated_rows.sort_values(by=['InvoiceNo','StockCode','Description','Quantity','CustomerID']) # sorting duplicate values

duplicated_rows_sorted.head(10)

df.duplicated().sum() # counting no of duplicated values

df.drop_duplicates(inplace=True) #dropping duplicate values

df.duplicated().sum() #rechecking on duplicate values

df.shape[0] # calculating no of rows

"""Handling Cancelled Transactions"""

df['Transaction_Status'] = np.where ( df['InvoiceNo'].astype(str).str.startswith('C') ,'Cancelled','Completed') # finding cancelled and completed transactions

cancelled_Transactions=df[df['Transaction_Status']== 'Cancelled'] # separating cancelled transactions
cancelled_Transactions.describe().drop('CustomerID', axis=1) # descrining cancelled transactions and dropping customerid as it is an identifier and not a numeric value

cancelled_percentage = (cancelled_Transactions.shape[0] / df.shape[0]) * 100 #Calculating precentage of cancelled transactions
cancelled_percentage

"""Handling unique Stockcode"""

unique_StockCodes= df['StockCode'].nunique() #finding unique stock codes
unique_StockCodes

top_10_stockCodes= df['StockCode'].value_counts(normalize=True).head(10) *100 #finding top 10 stockcodes

plt.figure(figsize=(12, 5)) #graph

plt.barh(top_10_stockCodes.index, top_10_stockCodes, color='#ff6200')

for i, val in enumerate(top_10_stockCodes):
    plt.text(val, i + 0.25, f'{val:.2f}%', fontsize=10)

plt.title('Top 10 Most Frequent Stock Codes')
plt.xlabel('Percentage Frequency (%)')
plt.ylabel('Stock Codes')
plt.gca().invert_yaxis()
plt.show()

unique_StockCodes = df['StockCode'].unique() #numeric char coint in stockcode

numeric_char_count_in_unique_StockCodes = pd.Series (unique_StockCodes).apply(lambda x : sum( c.isdigit() for c in str(x))).value_counts()

numeric_char_count_in_unique_StockCodes

anomalous_stock_codes = [code for code in unique_StockCodes if sum (c.isdigit() for c in str(code)) in (0,1)]

anomalous_stock_codes

for code in anomalous_stock_codes:
  print(code)

percentage_anomalous = (df ['StockCode'].isin(anomalous_stock_codes).sum()/ len(df)) *100

percentage_anomalous

df = df[~df['StockCode'].isin(anomalous_stock_codes)]

df.shape[0]

"""Handling descriptions"""

description_counts = df['Description'].value_counts() # finding and plotting 30 most frequent descriptions

top_30_descriptions = description_counts[:30]

plt.figure(figsize = (12,8))
plt.barh(top_30_descriptions.index[::-1], top_30_descriptions.values[::-1], color='#ff6200')


plt.xlabel('Number of Occurences')
plt.ylabel('Description')
plt.title('Top 30 Most Frequent Descriptions')
plt.show()

lowercase_descriptions = df['Description'].unique() # finding descriptions with lowercase letters
lowercase_descriptions = [desc for desc in lowercase_descriptions if any(char.islower() for char in desc)]
for desc in lowercase_descriptions:
    print(desc)

service_related_descriptions = ["Next Day Carriage", "High Resolution Image"] # finding percentage of service related percentage

service_related_percentage = df[df['Description'].isin(service_related_descriptions)].shape[0] / df.shape[0] * 100

service_related_percentage

df = df[~df['Description'].isin(service_related_descriptions)] # Converting all descriptions to uppercase

df['Description'] = df['Description'].str.upper()

df.shape[0]

df['UnitPrice'].describe()

df[df['UnitPrice']==0].describe()[['Quantity']]

df = df[df['UnitPrice'] > 0]

print(df[df['UnitPrice']<0])

df.reset_index(drop=True, inplace=True)

df.shape

"""**Feature Engineering**"""

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['InvoiceDay'] = df['InvoiceDate'].dt.date

customer_data = df.groupby('CustomerID')['InvoiceDay'].max().reset_index()

most_recent_date = df['InvoiceDay'].max()

customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
most_recent_date = pd.to_datetime(most_recent_date)

customer_data['Days_Since_Last_Purchase'] = (most_recent_date - customer_data['InvoiceDay']).dt.days

customer_data.drop(columns=['InvoiceDay'], inplace=True)

customer_data.head()

total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
total_transactions.rename(columns={'InvoiceNo': 'Total_Transactions'}, inplace=True)

total_products_purchased = df.groupby('CustomerID')['Quantity'].sum().reset_index()
total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

customer_data = pd.merge(customer_data, total_transactions, on='CustomerID')
customer_data = pd.merge(customer_data, total_products_purchased, on='CustomerID')

customer_data.head()

df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
total_spend = df.groupby('CustomerID')['Total_Spend'].sum().reset_index()

average_transaction_value = total_spend.merge(total_transactions, on='CustomerID')
average_transaction_value['Average_Transaction_Value'] = average_transaction_value['Total_Spend'] / average_transaction_value['Total_Transactions']

customer_data = pd.merge(customer_data, total_spend, on='CustomerID')
customer_data = pd.merge(customer_data, average_transaction_value[['CustomerID', 'Average_Transaction_Value']], on='CustomerID')

customer_data.head().T

unique_products_purchased = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
unique_products_purchased.rename(columns={'StockCode': 'Unique_Products_Purchased'}, inplace=True)

customer_data = pd.merge(customer_data, unique_products_purchased, on='CustomerID')

customer_data.head().T

df['Day_Of_Week'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour

days_between_purchases = df.groupby('CustomerID')['InvoiceDay'].apply(lambda x: (x.diff().dropna()).apply(lambda y: y.days))
average_days_between_purchases = days_between_purchases.groupby('CustomerID').mean().reset_index()
average_days_between_purchases.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)

favorite_shopping_day = df.groupby(['CustomerID', 'Day_Of_Week']).size().reset_index(name='Count')
favorite_shopping_day = favorite_shopping_day.loc[favorite_shopping_day.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Day_Of_Week']]

favorite_shopping_hour = df.groupby(['CustomerID', 'Hour']).size().reset_index(name='Count')
favorite_shopping_hour = favorite_shopping_hour.loc[favorite_shopping_hour.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Hour']]

customer_data = pd.merge(customer_data, average_days_between_purchases, on='CustomerID')
customer_data = pd.merge(customer_data, favorite_shopping_day, on='CustomerID')
customer_data = pd.merge(customer_data, favorite_shopping_hour, on='CustomerID')

customer_data.head().T

df['Country'].value_counts(normalize=True).head()

customer_country = df.groupby(['CustomerID', 'Country']).size().reset_index(name='Number_of_Transactions')

customer_main_country = customer_country.sort_values('Number_of_Transactions', ascending=False).drop_duplicates('CustomerID')

customer_main_country['Is_UK'] = customer_main_country['Country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)

customer_data = pd.merge(customer_data, customer_main_country[['CustomerID', 'Is_UK']], on='CustomerID', how='left')

customer_data.head().T

customer_data['Is_UK'].value_counts()

total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
cancellation_frequency = cancelled_transactions.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
cancellation_frequency.rename(columns={'InvoiceNo': 'Cancellation_Frequency'}, inplace=True)

customer_data = pd.merge(customer_data, cancellation_frequency, on='CustomerID', how='left', suffixes=('', '_y'))

customer_data['Cancellation_Frequency'].fillna(0, inplace=True)

customer_data['Cancellation_Rate'] = customer_data['Cancellation_Frequency'] / total_transactions['InvoiceNo']

customer_data.head().T

df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

monthly_spending = df.groupby(['CustomerID', 'Year', 'Month'])['Total_Spend'].sum().reset_index()

seasonal_buying_patterns = monthly_spending.groupby('CustomerID')['Total_Spend'].agg(['mean', 'std']).reset_index()
seasonal_buying_patterns.rename(columns={'mean': 'Monthly_Spending_Mean', 'std': 'Monthly_Spending_Std'}, inplace=True)

seasonal_buying_patterns['Monthly_Spending_Std'].fillna(0, inplace=True)

def calculate_trend(spend_data):
    if len(spend_data) > 1:
        x = np.arange(len(spend_data))
        slope, _, _, _, _ = linregress(x, spend_data)
        return slope
    else:
        return 0

spending_trends = monthly_spending.groupby('CustomerID')['Total_Spend'].apply(calculate_trend).reset_index()
spending_trends.rename(columns={'Total_Spend': 'Spending_Trend'}, inplace=True)

customer_data = pd.merge(customer_data, seasonal_buying_patterns, on='CustomerID')
customer_data = pd.merge(customer_data, spending_trends, on='CustomerID')

customer_data.head().T

customer_data['CustomerID'] = customer_data['CustomerID'].astype(str)

customer_data = customer_data.convert_dtypes()

customer_data.head(10).T

customer_data.info()

import pandas as pd
from sklearn.ensemble import IsolationForest

for col in ['Days_Since_Last_Purchase', 'Average_Days_Between_Purchases']:
    customer_data[col] = pd.to_numeric(customer_data[col], errors='coerce')

customer_data.fillna(0, inplace=True)

model = IsolationForest(contamination=0.05, random_state=0)

customer_data['Outlier_Scores'] = model.fit_predict(customer_data.iloc[:, 1:].to_numpy())

customer_data['Is_Outlier'] = [1 if x == -1 else 0 for x in customer_data['Outlier_Scores']]

customer_data.head().T

outlier_percentage = customer_data['Is_Outlier'].value_counts(normalize=True) * 100

plt.figure(figsize=(12, 4))
outlier_percentage.plot(kind='barh', color='#ff6200')

for index, value in enumerate(outlier_percentage):
    plt.text(value, index, f'{value:.2f}%', fontsize=15)

plt.title('Percentage of Inliers and Outliers')
plt.xticks(ticks=np.arange(0, 115, 5))
plt.xlabel('Percentage (%)')
plt.ylabel('Is Outlier')
plt.gca().invert_yaxis()
plt.show()

outliers_data = customer_data[customer_data['Is_Outlier'] == 1]

customer_data_cleaned = customer_data[customer_data['Is_Outlier'] == 0]

customer_data_cleaned = customer_data_cleaned.drop(columns=['Outlier_Scores', 'Is_Outlier'])

customer_data_cleaned.reset_index(drop=True, inplace=True)

customer_data_cleaned.shape[0]

sns.set_style('whitegrid')

corr = customer_data_cleaned.drop(columns=['CustomerID']).corr()

colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']
my_cmap = LinearSegmentedColormap.from_list('custom_map', colors, N=256)

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True

plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, cmap=my_cmap, annot=True, center=0, fmt='.2f', linewidths=2)
plt.title('Correlation Matrix', fontsize=14)
plt.show()

"""**Data Processing**"""

scaler = StandardScaler()

columns_to_exclude = ['CustomerID', 'Is_UK', 'Day_Of_Week']

columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)

customer_data_scaled = customer_data_cleaned.copy()

customer_data_scaled[columns_to_scale] = scaler.fit_transform(customer_data_scaled[columns_to_scale])

customer_data_scaled.head().T

customer_data_scaled.set_index('CustomerID', inplace=True)

pca = PCA().fit(customer_data_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

optimal_k = 6

sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

plt.figure(figsize=(20, 10))

barplot = sns.barplot(x=list(range(1, len(cumulative_explained_variance) + 1)),
                      y=explained_variance_ratio,
                      color='#fcc36d',
                      alpha=0.8)

lineplot, = plt.plot(range(0, len(cumulative_explained_variance)), cumulative_explained_variance,
                     marker='o', linestyle='--', color='#ff6200', linewidth=2)

optimal_k_line = plt.axvline(optimal_k - 1, color='red', linestyle='--', label=f'Optimal k value = {optimal_k}')

plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Explained Variance', fontsize=14)
plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

plt.xticks(range(0, len(cumulative_explained_variance)))
plt.legend(handles=[barplot.patches[0], lineplot, optimal_k_line],
           labels=['Explained Variance of Each Component', 'Cumulative Explained Variance', f'Optimal k value = {optimal_k}'],
           loc=(0.62, 0.1),
           frameon=True,
           framealpha=1.0,
           edgecolor='#ff6200')

x_offset = -0.3
y_offset = 0.01
for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
    plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
    if i > 0:
        plt.text(i + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)

plt.grid(axis='both')
plt.show()

pca = PCA(n_components=6)

customer_data_pca = pca.fit_transform(customer_data_scaled)

customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

customer_data_pca.index = customer_data_scaled.index

customer_data_pca.head()

def highlight_top3(column):
    top3 = column.abs().nlargest(3).index
    return ['background-color:  #57574f' if i in top3 else '' for i in column.index]
pc_df = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(pca.n_components_)])
                    # index=customer_data_scaled.columns)

pc_df.style.apply(highlight_top3, axis=0)

"""**K-Means Clustering**"""

sns.set(style='darkgrid', rc={'axes.facecolor': '#fcf0dc'}) #setting background color and grid color

sns.set_palette(['#ff6200']) #orange color for plots

km = KMeans(init='k-means++', n_init=10, max_iter=100, random_state=0)

fig, ax = plt.subplots(figsize=(12, 5))

visualizer = KElbowVisualizer(km, k=(2, 15), timings=False, ax=ax)

visualizer.fit(customer_data_pca)

visualizer.show();

def silhouette_analysis(df, start_k, stop_k, figsize=(15, 16)):
    """
    Perform Silhouette analysis for a range of k values and visualize the results.
    """

    plt.figure(figsize=figsize)

    grid = gridspec.GridSpec(stop_k - start_k + 1, 2)#creates grid layout for visuliazation

    first_plot = plt.subplot(grid[0, :])

    sns.set_palette(['darkorange'])

    silhouette_scores = []

    for k in range(start_k, stop_k + 1): #for calculating silhouette score
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=0)
        km.fit(df)
        labels = km.predict(df)
        score = silhouette_score(df, labels)
        silhouette_scores.append(score)

    best_k = start_k + silhouette_scores.index(max(silhouette_scores))#finding the best K

    plt.plot(range(start_k, stop_k + 1), silhouette_scores, marker='o')#plotting silhouette scores
    plt.xticks(range(start_k, stop_k + 1))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette score')
    plt.title('Average Silhouette Score for Different k Values', fontsize=15)

    optimal_k_text = f'The k value with the highest Silhouette score is: {best_k}'#giving the best value of k like actually displaying it
    plt.text(10, 0.23, optimal_k_text, fontsize=12, verticalalignment='bottom',
             horizontalalignment='left', bbox=dict(facecolor='#fcc36d', edgecolor='#ff6200', boxstyle='round, pad=0.5'))

    colors = sns.color_palette("bright")#setting bright color palatte

    for i in range(start_k, stop_k + 1):#cluster wise silhouette plotting
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
        row_idx, col_idx = divmod(i - start_k, 2)

        ax = plt.subplot(grid[row_idx + 1, col_idx])

        visualizer = SilhouetteVisualizer(km, colors=colors, ax=ax)#using silhouette visualizer from yellowbrick
        visualizer.fit(df)

        score = silhouette_score(df, km.labels_)# adding score to each plot
        ax.text(0.97, 0.02, f'Silhouette Score: {score:.2f}', fontsize=12, \
                ha='right', transform=ax.transAxes, color='red')

        ax.set_title(f'Silhouette Plot for {i} Clusters', fontsize=15)

    plt.tight_layout()
    plt.show()

silhouette_analysis(customer_data_pca, 3, 12, figsize=(20, 50))#calling the function

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=0)# applying k-means clustering
kmeans.fit(customer_data_pca)

cluster_frequencies = Counter(kmeans.labels_) #gives cluster no of each customer

label_mapping = {label: new_label for new_label, (label, _) in enumerate(cluster_frequencies.most_common())} # relabelling clusters

label_mapping = {v: k for k, v in {2: 1, 1: 0, 0: 2}.items()}#remapping clusters

new_labels = np.array([label_mapping[label] for label in kmeans.labels_])# assigning the new labels

customer_data_cleaned['cluster'] = new_labels#adding clusters to cleaned dataset

#naming for downloaded cleaned and clustered csv
cluster_map = {
    0: 'High Value Customers',
    1: 'Medium Value Customers',
    2: 'Low Value Customers'
}
customer_data_cleaned['Segment'] = customer_data_cleaned['cluster'].map(cluster_map)

customer_data_cleaned.to_csv('ecommerce_clustered.csv', index=False)

# downloading clean and clustered dataset
customer_data_cleaned.to_csv('output/ecommerce_clustered.csv', index=False)  # Create an `output/` folder first.


customer_data_pca['cluster'] = new_labels#saving labels in PCA data

customer_data_cleaned.head()

"""**Clustering Evaluation**"""

colors = ['#e8000b', '#1ac938', '#023eff']

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'colab'

cluster_0 = customer_data_pca[customer_data_pca['cluster'] == 0]#creating dataframes to hold customer from that particular cluster
cluster_1 = customer_data_pca[customer_data_pca['cluster'] == 1]
cluster_2 = customer_data_pca[customer_data_pca['cluster'] == 2]

colors = ['red', 'blue', 'green']
fig = go.Figure()#creating a 3D figure

fig.add_trace(go.Scatter3d(x=cluster_0['PC1'], y=cluster_0['PC2'], z=cluster_0['PC3'],# plotting the 3D PCA-transformed points for that cluster.
                           mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Cluster 0'))
fig.add_trace(go.Scatter3d(x=cluster_1['PC1'], y=cluster_1['PC2'], z=cluster_1['PC3'],
                           mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Cluster 1'))
fig.add_trace(go.Scatter3d(x=cluster_2['PC1'], y=cluster_2['PC2'], z=cluster_2['PC3'],
                           mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Cluster 2'))

fig.update_layout(                                                               #updating features for 3D figure
    title=dict(text='3D Visualization of Customer Clusters in PCA Space', x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC1'),
        yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC2'),
        zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC3'),
    ),
    width=900,
    height=800
)

fig.show()

cluster_percentage = (customer_data_pca['cluster'].value_counts(normalize=True) * 100).reset_index()#perc for cust per cluster
cluster_percentage.columns = ['Cluster', 'Percentage'] #renaming columns
cluster_percentage.sort_values(by='Cluster', inplace=True)

plt.figure(figsize=(10, 4))
sns.barplot(x='Percentage', y='Cluster', data=cluster_percentage, orient='h', palette=colors)

for index, value in enumerate(cluster_percentage['Percentage']):#adds percentages at the end of each bar
    plt.text(value+0.5, index, f'{value:.2f}%')

plt.title('Distribution of Customers Across Clusters', fontsize=14)
plt.xticks(ticks=np.arange(0, 50, 5))
plt.xlabel('Percentage (%)')

plt.show()

num_observations = len(customer_data_pca)

X = customer_data_pca.drop('cluster', axis=1)
clusters = customer_data_pca['cluster']

sil_score = silhouette_score(X, clusters)
calinski_score = calinski_harabasz_score(X, clusters)
davies_score = davies_bouldin_score(X, clusters)

table_data = [
    ["Number of Observations", num_observations],
    ["Silhouette Score", sil_score],
    ["Calinski Harabasz Score", calinski_score],
    ["Davies Bouldin Score", davies_score]
]

print(tabulate(table_data, headers=["Metric", "Value"], tablefmt='pretty'))

"""**Cluster Analysis**"""

df_customer = customer_data_cleaned.set_index('CustomerID')

scaler = StandardScaler()#standardizing numerical features
df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['cluster', 'Segment'], axis=1))

df_customer_standardized = pd.DataFrame(df_customer_standardized, columns=df_customer.columns[:-2], index=df_customer.index) # Aadding bcak features
df_customer_standardized['cluster'] = df_customer['cluster']
df_customer_standardized['Segment'] = df_customer['Segment']

cluster_centroids = df_customer_standardized.drop(columns=['Segment']).groupby('cluster').mean()# calac mean value of each feature for each cluster

def create_radar_chart(ax, angles, data, color, cluster): #setting up for radar chart

    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')


    ax.set_title(f'Cluster {cluster}', size=20, color=color, y=1.1)

labels=np.array(cluster_centroids.columns)#setting axis
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
labels = np.concatenate((labels, [labels[0]]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(polar=True), nrows=1, ncols=3)#plotting radar chart for each cluster

for i, color in enumerate(colors):
    data = cluster_centroids.loc[i].tolist()
    data += data[:1]
    create_radar_chart(ax[i], angles, data, color, i)

ax[0].set_xticks(angles[:-1])
ax[0].set_xticklabels(labels[:-1])

ax[1].set_xticks(angles[:-1])
ax[1].set_xticklabels(labels[:-1])

ax[2].set_xticks(angles[:-1])
ax[2].set_xticklabels(labels[:-1])

ax[0].grid(color='grey', linewidth=0.5)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

colors = plt.cm.tab10.colors

features = customer_data_cleaned.columns[1:-1]#excluding first and last feature
clusters = customer_data_cleaned['cluster'].unique()#sorting unique clusteirng id
clusters.sort()

n_rows = len(features)#creating grid
n_cols = len(clusters)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))

for i, feature in enumerate(features):#preparing clean array values to pass
    for j, cluster in enumerate(clusters):
        data = customer_data_cleaned[customer_data_cleaned['cluster'] == cluster][feature]

        data = data.to_numpy().flatten()


        axes[i, j].hist(data, bins=20, color=colors[j], edgecolor='w', alpha=0.7)

        axes[i, j].set_title(f'Cluster {cluster} - {feature}', fontsize=15)
        axes[i, j].set_xlabel('')
        axes[i, j].set_ylabel('')


plt.tight_layout()
plt.show()

"""**Recommendation System**"""

outlier_customer_ids = outliers_data['CustomerID'].astype('float').unique()
df_filtered = df[~df['CustomerID'].isin(outlier_customer_ids)]

customer_data_cleaned['CustomerID'] = customer_data_cleaned['CustomerID'].astype('float')

merged_data = df_filtered.merge(customer_data_cleaned[['CustomerID', 'cluster']], on='CustomerID', how='inner')

best_selling_products = merged_data.groupby(['cluster', 'StockCode', 'Description'])['Quantity'].sum().reset_index()
best_selling_products = best_selling_products.sort_values(by=['cluster', 'Quantity'], ascending=[True, False])
top_products_per_cluster = best_selling_products.groupby('cluster').head(10)

customer_purchases = merged_data.groupby(['CustomerID', 'cluster', 'StockCode'])['Quantity'].sum().reset_index()

recommendations = []
for cluster in top_products_per_cluster['cluster'].unique():
    top_products = top_products_per_cluster[top_products_per_cluster['cluster'] == cluster]
    customers_in_cluster = customer_data_cleaned[customer_data_cleaned['cluster'] == cluster]['CustomerID']

    for customer in customers_in_cluster:

        customer_purchased_products = customer_purchases[(customer_purchases['CustomerID'] == customer) &
                                                         (customer_purchases['cluster'] == cluster)]['StockCode'].tolist()

        top_products_not_purchased = top_products[~top_products['StockCode'].isin(customer_purchased_products)]
        top_3_products_not_purchased = top_products_not_purchased.head(3)


        recommendations.append([customer, cluster] + top_3_products_not_purchased[['StockCode', 'Description']].values.flatten().tolist())


recommendations_df = pd.DataFrame(recommendations, columns=['CustomerID', 'cluster', 'Rec1_StockCode', 'Rec1_Description', \
                                                 'Rec2_StockCode', 'Rec2_Description', 'Rec3_StockCode', 'Rec3_Description'])
customer_data_with_recommendations = customer_data_cleaned.merge(recommendations_df, on=['CustomerID', 'cluster'], how='right')

customer_data_with_recommendations.set_index('CustomerID').iloc[:, -6:].sample(10, random_state=0)