import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random

plt.ion()

dataset_2022_path_410 = r'data\clean_data\cleaned_data_with_deltavalues_2022_line410.xlsx'
dataset = r'data\clean_data\cleaned_data_with_deltavalues_2022_2023_2024.xlsx'

df2022 = pd.read_excel(dataset_2022_path_410)
df_all = pd.read_excel(dataset)

df = pd.concat([df2022, df_all], axis=0)
df = df.drop("Recording Date", axis=1)

# Remove rows where "Defect Code" is equal to 0
df = df[df["Defect Code"] != 0]

# top_defects = [27, 134, 105, 106, 29]
# # top_defects = [4, 14, 27, 29, 100, 104, 105, 106, 132, 134]
# df = df[df["Defect Code"].isin(top_defects)]

# Features
x = df.drop(["Defect Code", "Group"], axis=1)  # Drop both "Defect Code" and "Group" columns
x = StandardScaler().fit_transform(x)

# Target
y = df["Group"]  # Use the existing "Group" column as the target

# PCA analysis

# # Perform PCA
# pca = PCA()
# pca.fit(x)
#
# # Plot explained variance ratio
# plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Scree Plot')
# plt.show()

pca = PCA(n_components=7)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents, columns=[f'principal component {i+1}' for i in range(7)])

df.reset_index(drop=True, inplace=True)
finalDf = pd.concat([principalDf, df[['Group']]], axis=1)

fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_title('3 component PCA by Group', fontsize=15)

plt.xlabel('Principal Component 1', fontsize=15)
plt.ylabel('Principal Component 2', fontsize=15)
plt.title('2D PCA by Group', fontsize=15)

targets = finalDf["Group"].unique()

colors = {}
for target in targets:
    colors[target] = "#" + "%06x" % random.randint(0, 0xFFFFFF)

for target in targets:
    indicesToKeep = finalDf['Group'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                finalDf.loc[indicesToKeep, 'principal component 2'],
                c=colors[target],
                s=50,
                label=target)

plt.legend(targets)
plt.grid(True)
plt.savefig(r'plots\pca\pca_2d_by_group.png')
plt.show()

# #########################
# # Analysis by Defect Code
# #########################

# # Features
# x = df.drop("Defect Code", axis=1)
# x = StandardScaler().fit_transform(x)
#
# # Target
# y = df["Defect Code"]
#
# # PCA analysis
#
# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(x)
#
# # principalDf = pd.DataFrame(data=principalComponents,
# #                            columns=['principal component 1', 'principal component 2'])
#
# principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'
#     , 'principal component 3'])
#
# finalDf = pd.concat([principalDf, df[['Defect Code']].reset_index(drop=True)], axis=1)
#
# # fig = plt.figure(figsize=(8, 8))
# # ax = fig.add_subplot(1, 1, 1)
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_zlabel('Principal Component 3', fontsize=10)
# ax.set_title('3 component PCA', fontsize=15)
#
# # ax.set_title('2 component PCA', fontsize=20)
#
# targets = finalDf["Defect Code"].unique()
#
# # Random colors for each defect code
# colors = {}
# for target in targets:
#     colors[target] = "#" + "%06x" % random.randint(0, 0xFFFFFF)
#
# for target in targets:
#     indicesToKeep = finalDf['Defect Code'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
#                finalDf.loc[indicesToKeep, 'principal component 2'],
#                finalDf.loc[indicesToKeep, 'principal component 3'],
#                c=colors[target],
#                s=50,
#                label=target)
#
# ax.legend(targets)
# ax.grid(True)
# plt.savefig(r'plots\pca\pca_3d_top5defects.png')
# plt.show()
