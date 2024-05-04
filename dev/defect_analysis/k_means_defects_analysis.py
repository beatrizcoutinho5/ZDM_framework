import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

train_datasets = r'..\data\clean_data\cleaned_data_2022_line410.xlsx'
test_dataset = r'..\data\clean_data\cleaned_data2022_2023_2024.xlsx'

df2022 = pd.read_excel(train_datasets)
df_all = pd.read_excel(test_dataset)

df = pd.concat([df2022, df_all], axis=0)
df = df.drop("Recording Date", axis=1)
df = df[df["Defect Code"] != 0]

# Features
x = df.drop(["Defect Code", "Group"], axis=1)
x = StandardScaler().fit_transform(x)

# Target
y = df["Group"]

# K Means
clusters = [3, 5, 10, 15]

for number_of_clusters in clusters:

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)
    kmeans.fit(x)
    df['Cluster'] = kmeans.labels_

    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 0], x[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.5)
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.savefig(r'k_means_' + str(number_of_clusters) + '_clusters.png')
    plt.show()

    cluster_counts = df.groupby(['Defect Code', 'Cluster']).size().unstack(fill_value=0)
    defect_code_percentages = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100

    defect_code_percentages.to_excel(str(number_of_clusters) + 'cluster_distribution.xlsx')



