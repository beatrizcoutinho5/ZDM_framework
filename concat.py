import pandas as pd


df2022 = pd.read_excel(r'data\clean_data\cleaned_data_with_deltavalues_2022_line410.xlsx')
df2023 = pd.read_excel(r'data\clean_data\cleaned_data_with_deltavalues_2023_line410.xlsx')
df2024 = pd.read_excel(r'data\clean_data\cleaned_data_with_deltavalues_2024_line410.xlsx')


df = pd.concat([df2022, df2023, df2024], axis=0)
df.to_excel(r'data\clean_data\cleaned_data_with_deltavalues2022_2023_2024.xlsx', index=False)

print("Saved!")