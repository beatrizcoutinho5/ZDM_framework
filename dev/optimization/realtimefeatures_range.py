import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = r'../data/clean_data/binary_cleaned_data2022_2023_2024.xlsx'
data_path_delta = r'../data/clean_data/cleaned_data_with_deltavalues2022_2023_2024.xlsx'

plots_dir = '../plots/real_time_features_range'

df = pd.read_excel(data_path)
df_delta = pd.read_excel(data_path_delta)

df = df[df['Defect Code'] == 0]
df_delta = df_delta[df_delta['Defect Code'] == 0]

real_time_features = ['Mechanical Cycle Time', 'Cycle Time', 'Carriage Speed']

real_time_features_delta = ['Thermal Cycle Time', 'Pressure',
                            'Upper Plate Temperature', 'Lower Plate Temperature',
                            'Press Input Table Speed', 'Scraping Cycle', 'Transverse Saw Cycle']

for feature in real_time_features:

    if (feature == 'Mechanical Cycle Time') or (feature == 'Cycle Time'):

        values = df[feature]
        plt.hist(values, bins=np.arange(0, 1001, 50), color='orange')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature}')
        plt.grid(True)
        plt.savefig(f'{plots_dir}/{feature}')

    else:

        values = df[feature]
        plt.hist(values, bins='auto', color='orange')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature}')
        plt.grid(True)
        plt.savefig(f'{plots_dir}/{feature}')

for feature in real_time_features_delta:

    values = df_delta[feature]
    plt.hist(values, bins=np.arange(0, 1001, 50), color='orange')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    plt.grid(True)
    plt.savefig(f'{plots_dir}/{feature}')
