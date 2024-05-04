import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram

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

# hierarchical clustering

linkage_data = linkage(x, method='ward', metric='euclidean')
plt.figure(figsize=(20, 10))

dendrogram(linkage_data, truncate_mode='lastp', p=4)

plt.show()

# height threshold for clustering
threshold = 300

clusters = fcluster(linkage_data, threshold, criterion='distance')

# DF combining original Group and cluster assignments
result_df = pd.DataFrame({'Group': df["Group"], 'Cluster': clusters})

# Occurrences of each group within each cluster
defect_counts = result_df.groupby(['Cluster', 'Group']).size().unstack(fill_value=0)

print("Number of each Group assigned to each Cluster:")
print(defect_counts)
