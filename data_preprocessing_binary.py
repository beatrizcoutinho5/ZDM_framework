import pandas as pd

datasets = [r'data\clean_data\cleaned_data_2022_line410.xlsx', r'data\clean_data\cleaned_data2022_2023_2024.xlsx' ]
for data in datasets:

    df = pd.read_excel(data)

    df['Defect Code'] = df['Defect Code'].apply(lambda x: 1 if x != 0 else 0)

    if data == r'data\clean_data\cleaned_data_2022_line410.xlsx':
        df.to_excel(r'data\clean_data\binary_cleaned_data_2022_line410.xlsx', index=False)
    elif data == r'data\clean_data\cleaned_data2022_2023_2024.xlsx':
        df.to_excel(r'data\clean_data\binary_cleaned_data2022_2023_2024.xlsx', index=False)
