import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.graphics.tsaplots import plot_acf

# Directory to save the resulting plots
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

dataset_2022_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\dataset_jan_dec_2022_line410.xlsx'
dataset_2023_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\dataset_jan_may_2023_line410.xlsx'
dataset_2024_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\dataset_jan_feb_2024_line410.xlsx'

df2022 = pd.read_excel(dataset_2022_path)
df2023 = pd.read_excel(dataset_2023_path)

df = pd.concat([df2022, df2023], axis=0)

# Make plots and prints?
make_plots = 1

# Autocorrelation and Time Series Analysis?
autocorrelation_analysis = 1

# Print initial dataset characteristics
initial_number_features = len(df.columns)
initial_number_samples = len(df)

print(f"Initial Number of Features: {initial_number_features}")
print(f"Initial Number of Samples: {initial_number_samples}")

# Remove rows with Line Working? = -1
df = df[df["Line Working?"] != -1]

# Remove rows Defect Code = -1
df = df[df["Defect Code"] != -1]

# Remove columns with irrelevant data for the framework
columns_remove = ["Defect Group", "Defect Group Description", "Defect Description", "Pallet Code",
                  "Pallet Code Production Date", "Line Working?", "Humidity", "Temperature",
                  "Calculated Thermal Cycle Time", "Single Station Thermal Cycle Time",
                  "Double Station Thermal Cycle Time", "Jaw Clamping Time", "Suction Cup Pickup Time",
                  "Scratch Test"]

df = df.drop(columns_remove, axis=1)

# Transform categorical features into numerical representation

# features with few unique value were considered as categorical
columns_cat = ["GFFTT", "Finishing Top", "Finishing Down", "Reference Top", "Reference Down"]

for i in columns_cat:
    # categorical representation
    df[i] = df[i].astype('category')
    # new column numerical code for every category in the feature
    df[i + '_cat'] = df[i].cat.codes

df = df.drop(columns_cat, axis=1)

# Remove rows with any missing values
df = df.dropna(how='any', axis=0)

# Remove feature with only one unique value
df = df.loc[:, df.apply(pd.Series.nunique) > 1]

# Pearson Correlation factor

numeric_columns = df.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()

if make_plots == 1:
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(12, 8))
    heatmap_plot = sns.heatmap(correlation_matrix, mask=mask, center=0, annot=False, fmt='.2f', square=True,
                               cmap='coolwarm')

    pearson_subdirectory_path = os.path.join(plots_dir, 'pearson_correlation')
    os.makedirs(pearson_subdirectory_path, exist_ok=True)

    output_path = os.path.join(pearson_subdirectory_path, 'pearson_correlation_matrix.png')
    plt.savefig(output_path, bbox_inches='tight')

threshold = 0.9

if make_plots == 1:

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                print(
                    f" {correlation_matrix.columns[i]}, {correlation_matrix.columns[j]} , {correlation_matrix.iloc[i, j]:.2f}")

columns_remove_correlated = ["Thickness.1", "Lower Valve Temperature", "Upper Valve Temperature", "Liston 1 Speed.1",
                             "Liston 2 Speed.1", "Floor 1.1", "Floor 2.1", "Bridge Platform.1", "Floor 1 Blow Time.1",
                             "Floor 2 Blow Time.1", "Centering Table.1", "Finishing Down_cat", "Reference Down_cat"]

# df = df.drop(columns_remove_correlated, axis=1)
for col in columns_remove_correlated:
    if col in df.columns:
        df = df.drop(col, axis=1)

###################################
# Data Analysis and Visualization #
###################################

categorical_features = ["Production Order Code", "Production Order Opening", "Defect Code",
                        "Control Panel with Micro Stop", "Floor 1", "Floor 2", "Bridge Platform", "Floor 1 Blow Time",
                        "Floor 2 Blow Time", "Left Jaw Discharge", "GFFTT_cat", "Finishing Top_cat"]

continuous_features = ["Quantity", "Length", "Width", "Thickness", "Lot Size", "Cycle Time", "Mechanical Cycle Time",
                       "Thermal Cycle Time", "Control Panel Delay Time", "Sandwich Preparation Time", "Carriage Time",
                       "Lower Plate Temperature", "Upper Plate Temperature", "Pressure", "Liston 1 Speed",
                       "Liston 2 Speed", "Centering Table", "Carriage Speed", "Take-off Path", "Take-off Time",
                       "High Pressure Input Time", "Press Input Table Speed", "Paper RC", "Paper VC",
                       "Paper Shelf Life", "Reference Top_cat"]

# Width as assigned as a 'Object' type, so it needed to be typecasted into a numeric representation
df['Width'] = pd.to_numeric(df['Width'], errors='coerce')
df = df.dropna(subset=['Width'])
df['Width'] = df['Width'].astype(int)

# Plotting box plots for the continuous features to identify outliers

if make_plots == 1:

    boxplot_subdirectory_path = os.path.join(plots_dir, 'boxplots')
    os.makedirs(boxplot_subdirectory_path, exist_ok=True)

    for i, feature in enumerate(continuous_features, 1):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Values')

        output_path = os.path.join(boxplot_subdirectory_path, f'{feature}_box_plot.png')
        plt.savefig(output_path, bbox_inches='tight')

        plt.close()

    print(f"Individual box plots saved.")

# Removing zero values (if they exist) from the specified features

features_remove_zero = ['Carriage Speed', 'High Pressure Input Time', 'Liston 1 Speed', 'Liston 2 Speed', 'Lot Size',
                        'Mechanical Cycle Time', 'Press Input Table Speed', 'Pressure', 'Quantity',
                        'Carriage Time', 'Centering Table', 'Lower Plate Temperature', 'Upper Plate Temperature']

rows_zero_removed_count = 0

for col in features_remove_zero:
    rows_to_remove = df[df[col] <= 0]
    rows_zero_removed_count += len(rows_to_remove)

# Remove the rows with values less than or equal to 0 in any of the specified columns
df = df.drop(rows_to_remove.index, axis=0)

df = df[(df[features_remove_zero] > 0).all(axis=1)]

# Print initial dataset characteristics

print()

final_number_features = len(df.columns)
final_number_samples = len(df)

print(f"Final Number of Features: {final_number_features}")
print(f"Final Number of Samples: {final_number_samples}")

print()
print(f"Removed {np.abs(final_number_features - initial_number_features)} features.")
print(f"Removed {np.abs(final_number_samples - initial_number_samples)} samples.")

#####################################
# Feature Auto-correlation Analysis #
#####################################

if autocorrelation_analysis == 1:

    # Consider only features with numeric values
    numeric_columns = df.select_dtypes(include=['number'])

    # Dropping categorical features
    columns_to_drop = ['Production Order Code', 'Production Order Opening', 'GFFTT_cat', 'Finishing Top_cat',
                       'Reference Top_cat']

    numeric_columns = numeric_columns.drop(columns=columns_to_drop)

    # Calculate feature autocorrelation
    autocorr_results = numeric_columns.apply(lambda x: x.autocorr())

    # Filter columns with autocorrelation greater than or equal to 0.55
    high_autocorr_columns = autocorr_results[abs(autocorr_results) >= 0.55].index.tolist()

    if make_plots == 1:
        print("Columns with autocorrelation greater than or equal to |0.55|:")
        print(high_autocorr_columns)

    # ACF plots

    if make_plots == 1:

        acf_subdirectory_path = os.path.join(plots_dir, 'acf')
        os.makedirs(acf_subdirectory_path, exist_ok=True)

        cols = df.columns.tolist()
        cols.remove("Recording Date")

        for i, c in enumerate(cols):
            plt.figure(i)
            plt.clf()
            ax = plot_acf(df[c], title=c)
            plt.ylabel(c)

            output_path = os.path.join(acf_subdirectory_path, f'{c}_acf_plot.png')
            plt.savefig(output_path, bbox_inches='tight')

            plt.close('all')

        print("Autocorrelation plots saved.")

    for col in (high_autocorr_columns + ["Recording Date"] + ["Quantity"]):
        # Create a new column that contains the feature values shifting the value by one (lag=1)
        df["{}-1".format(col)] = df[col].shift(1)

    # Calculate the time difference between consecutive timestamps in minutes
    df["Delta Time"] = pd.to_timedelta(df["Recording Date"] - df["Recording Date-1"]).dt.total_seconds() / 60

    # Calculate the rate of change (Delta) for each feature using the lagged values and the Delta Time between the,
    for col in high_autocorr_columns:
        df["Delta_{}".format(col)] = (df[col] - df["{}-1".format(col)]) / df["Delta Time"]

    # Remove redundant features after the autocorrelation process

    for col in (high_autocorr_columns + ["Recording Date"] + ["Quantity"]):
        df = df.drop("{}-1".format(col), axis=1)

    df = df.dropna(axis=0)
    df.to_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\cleaned_data_with_deltavalues2022_2023.xlsx',
        index=False)
    print("Saved clean data!")

if autocorrelation_analysis == 0:
    df = df.dropna(axis=0)
    df.to_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\cleaned_data.xlsx',
        index=False)
