import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset_2022_path_410 = r'data\clean_data\cleaned_data_with_deltavalues_2022_line410.xlsx'
dataset = r'data\clean_data\cleaned_data_with_deltavalues2022_2023_2024.xlsx'

df2022 = pd.read_excel(dataset_2022_path_410)
df_all = pd.read_excel(dataset)

df = pd.concat([df2022, df_all], axis=0)
df = df.drop("Recording Date", axis=1)
df = df[df["Defect Code"] != 0]

# Features
x = df.drop(["Defect Code", "Group"], axis=1)
x = StandardScaler().fit_transform(x)

# Target
y = df["Group"]

# # K Means
# clusters = [15]
#
# for number_of_clusters in clusters:

number_of_clusters = 3

kmeans = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)
kmeans.fit(x)
df['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
plt.scatter(x[:, 0], x[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.savefig(r'plots\k_means\k_means_' + str(number_of_clusters) + '.png')
plt.show()

cluster_counts = df.groupby(['Defect Code', 'Cluster']).size().unstack(fill_value=0)
defect_code_percentages = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100

defect_code_percentages.to_excel('new_cluster_percentages_' + str(number_of_clusters) + '.xlsx')

# # hiearchical clustering
#
# linkage_data = linkage(x, method='ward', metric='euclidean')
# plt.figure(figsize=(20, 10))
#
# dendrogram(linkage_data, truncate_mode='lastp', p=4)
#
# plt.show()
#
# # Choose a height threshold for clustering
# threshold = 300  # Adjust this threshold as needed
#
# # Assign clusters based on the threshold
# clusters = fcluster(linkage_data, threshold, criterion='distance')
#
# # Create a DataFrame combining original Defect Code and cluster assignments
# result_df = pd.DataFrame({'Group': df["Group"], 'Cluster': clusters})
#
# # Count the occurrences of each Defect Code within each Cluster
# defect_counts = result_df.groupby(['Cluster', 'Group']).size().unstack(fill_value=0)
#
# print("Number of each Group assigned to each Cluster:")
# print(defect_counts)

