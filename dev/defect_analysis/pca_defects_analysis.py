import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.ion()

dataset_2022_path_410 = r'..\data\clean_data\cleaned_data_2022_line410.xlsx'
dataset = r'..\data\clean_data\cleaned_data2022_2023_2024.xlsx'

df2022 = pd.read_excel(dataset_2022_path_410)
df_all = pd.read_excel(dataset)

df = pd.concat([df2022, df_all], axis=0)
df = df.drop("Recording Date", axis=1)
df = df[df["Defect Code"] != 0]

# 1 -> analysis by defect group
# 0 -> analysis by defect code
defect_group_analysis = 1

if defect_group_analysis == 1:

    ###########################################################
    #### using the defect group instead of the defect code ####
    ###########################################################

    # Features
    x = df.drop(["Defect Code", "Group"], axis=1)
    x = StandardScaler().fit_transform(x)

    # Target
    y = df["Group"]

    # PCA analysis

    # # to choose the best number of components (looking at the scree plot)
    # pca = PCA()
    # pca.fit(x)
    #
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

    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.title('PCA by Group', fontsize=15)

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
    plt.savefig(r'..\plots\pca\pca_2d_by_group.png')
    plt.show()

###############################
#### using the defect code ####
###############################

else:

    top_defects = [27, 134, 105, 106, 29] # top 5 defects
    df = df[df["Defect Code"].isin(top_defects)]

    # Features
    x = df.drop(["Defect Code", "Group"], axis=1)
    x = StandardScaler().fit_transform(x)

    # Target
    y = df["Defect Code"]

    # PCA analysis

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)

    # principalDf = pd.DataFrame(data=principalComponents,
    #                            columns=['principal component 1', 'principal component 2'])

    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'
        , 'principal component 3'])

    finalDf = pd.concat([principalDf, df[['Defect Code']].reset_index(drop=True)], axis=1)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=10)
    ax.set_title('3 component PCA', fontsize=15)

    # ax.set_title('2 component PCA', fontsize=20)

    targets = finalDf["Defect Code"].unique()

    colors = {}
    for target in targets:
        colors[target] = "#" + "%06x" % random.randint(0, 0xFFFFFF)

    for target in targets:
        indicesToKeep = finalDf['Defect Code'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   finalDf.loc[indicesToKeep, 'principal component 3'],
                   c=colors[target],
                   s=50,
                   label=target)

    ax.legend(targets)
    ax.grid(True)
    plt.savefig(r'..\plots\pca\pca_3d_top5defects_by_code.png')
    plt.show()
