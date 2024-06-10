import pandas as pd

dataset1 = pd.read_excel(r'../data/dataset_jan_dec_2022_line409.xlsx')
dataset2 = pd.read_excel(r'../data/dataset_jan_dec_2022_line410.xlsx')
dataset3 = pd.read_excel(r'../data/dataset_jan_feb_2024_line410.xlsx')
dataset4 = pd.read_excel(r'../data/dataset_jan_may_2023_line410.xlsx')

dataset = pd.concat([dataset1, dataset2, dataset3, dataset4])

total_samples = dataset['Quantity'].sum()

dataset['Defect Code'] = dataset['Defect Code'].astype(str)
dataset_defects = dataset[dataset['Defect Code'] != '0']
defect_samples = dataset_defects['Quantity'].sum()

defect_percentage = round((defect_samples/total_samples)*100, 3)

print("Total Number of Samples: ", total_samples)
print("Total Number of Defects: ", defect_samples)
print("Percentage of Defects:", defect_percentage)
