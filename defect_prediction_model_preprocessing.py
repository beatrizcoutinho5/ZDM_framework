import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import os

clean_data_delta_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\cleaned_data_with_deltavalues.xlsx'

df = pd.read_excel(clean_data_delta_path)

# Directory to save the resulting plots
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Removing the date from the data

df = df.drop("Recording Date", axis=1)

# The top 10 defects make up almost 80% of all defect occurrence
# those defects are the only ones being considered in the classification process
# since the rest of the defects have limited samples/instances

defect_code_column = 'Defect Code'
quantity_column = 'Quantity'

defect_quantity_dict = {}

for index, row in df.iterrows():
    code = row[defect_code_column]
    quantity = row[quantity_column]

    defect_quantity_dict[code] = defect_quantity_dict.get(code, 0) + quantity

# # Remove the Defect Code = 0, that means that there is no defect
# defect_quantity_dict.pop(0, None)

# Sort by quantity
sorted_defect_quantity = sorted(defect_quantity_dict.items(), key=lambda x: x[1], reverse=True)

# Save the top 10 defect codes
top_defects = [code for code, quantity in sorted_defect_quantity[:11]]

# Remove the instances of other defects from the dataframe
df = df[df["Defect Code"].isin(top_defects)]

# Input values and targe features
y_values = df["Defect Code"]
x_values = df.drop("Defect Code", axis=1)

# Split train / test
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=None)

# Calculate the quantity of each defect code in the train set
defect_count = y_train.value_counts()
defect_count_without_zero = y_train[y_train != 0].value_counts()

# Data Augmentation using SMOTE

x_train_aug, y_train_aug = SMOTE(random_state=42).fit_resample(x_train, y_train)

defect_count_aug = y_train_aug.value_counts()
defect_count_without_zero_aug = y_train_aug[y_train_aug != 0].value_counts()

# Plot the frequency of defects before and after data augmentation

augmentation_subdirectory_path = os.path.join(plots_dir, 'augmentation')
os.makedirs(augmentation_subdirectory_path, exist_ok=True)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Plot the bar plot with Defect Code = 0 in the original data
sns.barplot(x=defect_count.index, y=defect_count.values, palette="magma", ax=axes[0, 0])
axes[0, 0].set_title("Original - Quantity of the selected defect codes in TRAIN set + No Defect")
axes[0, 0].set_xlabel("Defect Code")
axes[0, 0].set_ylabel("Quantity")

# Plot the bar plot with only defects (excluding Defect Code = 0) in the original data
sns.barplot(x=defect_count_without_zero.index, y=defect_count_without_zero.values, palette="viridis", ax=axes[0, 1])
axes[0, 1].set_title("Original - Quantity of the selected defect codes in TRAIN set (excluding code 0)")
axes[0, 1].set_xlabel("Defect Code")
axes[0, 1].set_ylabel("Quantity")

# Plot the bar plot with Defect Code = 0 in the augmented data
sns.barplot(x=defect_count_aug.index, y=defect_count_aug.values, palette="magma", ax=axes[1, 0])
axes[1, 0].set_title("AUG - Quantity of the selected defect codes in TRAIN set + No Defect")
axes[1, 0].set_xlabel("Defect Code")
axes[1, 0].set_ylabel("Quantity")

# Plot the bar plot with only defects (excluding Defect Code = 0) in the augmented data
sns.barplot(x=defect_count_without_zero_aug.index, y=defect_count_without_zero_aug.values, palette="viridis", ax=axes[1, 1])
axes[1, 1].set_title("AUG - Quantity of the selected defect codes in TRAIN set (excluding code 0)")
axes[1, 1].set_xlabel("Defect Code")
axes[1, 1].set_ylabel("Quantity")

output_path = os.path.join(augmentation_subdirectory_path, 'data_before_and_after_SMOTE.png')
plt.savefig(output_path)

plt.tight_layout()
plt.show()





