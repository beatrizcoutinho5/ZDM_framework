import pandas as pd

df = pd.read_excel(r'data\clean_data\cleaned_data_2022_line410.xlsx')
# df = pd.read_excel(r'data\clean_data\cleaned_data2022_2023_2024.xlsx')

df['Defect Code'] = df['Defect Code'].apply(lambda x: 1 if x != 0 else 0)

df.to_excel(r'data\clean_data\binary_cleaned_data_2022_line410.xlsx', index=False)
# df.to_excel(r'data\clean_data\binary_cleaned_data2022_2023_2024.xlsx', index=False)
