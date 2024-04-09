import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

# PCA to visualize data in 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

# K Means
clusters = [2, 3, 4]

for number_of_clusters in clusters:

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42, n_init = 10)
    kmeans.fit(x)
    df['Cluster'] = kmeans.labels_


    plt.figure(figsize=(10, 6))
    plt.scatter(principalDf['PC1'], principalDf['PC2'], c=df['Cluster'], cmap='viridis', alpha=0.5)
    plt.title('K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.savefig(r'plots\k_means\k_means_' + str(number_of_clusters) + '.png')
    plt.show()

    cluster_counts = df.groupby(['Defect Code', 'Cluster']).size().unstack(fill_value=0)
    defect_code_percentages = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100

    defect_code_percentages.to_excel('cluster_percentages_' + str(number_of_clusters) + '.xlsx')