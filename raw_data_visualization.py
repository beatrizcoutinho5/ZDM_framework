import pandas as pd
import matplotlib.pyplot as plt


dataset_file_paths = [
    r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\dataset_jan_dec_2022_line409.xlsx',
    r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\dataset_jan_dec_2022_line410.xlsx',
    r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\dataset_jan_may_2023_line410.xlsx',
]

dataset_2023_path= r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\dataset_jan_may_2023_line410.xlsx'

df= pd.read_excel(dataset_2023_path)

columns_remove = ["Defect Group", "Defect Group Description", "Defect Description", "Pallet Code",
                  "Pallet Code Production Date", "Line Working?", "Humidity", "Temperature",
                  "Calculated Thermal Cycle Time", "Single Station Thermal Cycle Time",
                  "Double Station Thermal Cycle Time"]

df = df.drop(columns_remove, axis=1)

# Print the number of unique values for every feature

for i, column in enumerate(df.columns):
    unique_count = df[column].nunique()
    print(f"{column}: {unique_count}")












# fig, axes = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(15, 5))
#
# for i, column in enumerate(df.columns):
#     axes[i].boxplot(df[column])
#     axes[i].set_title(column)
#
#
# plt.tight_layout()
# plt.show()


