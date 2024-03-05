import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

clean_data_delta_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\cleaned_data_with_deltavalues_2023.xlsx'

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

split_index = int(0.8 * len(df))

train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

y_train = train_data["Defect Code"]
x_train = train_data.drop("Defect Code", axis=1)

y_test = test_data["Defect Code"]
x_test = test_data.drop("Defect Code", axis=1)

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
sns.barplot(x=defect_count_without_zero_aug.index, y=defect_count_without_zero_aug.values, palette="viridis",
            ax=axes[1, 1])
axes[1, 1].set_title("AUG - Quantity of the selected defect codes in TRAIN set (excluding code 0)")
axes[1, 1].set_xlabel("Defect Code")
axes[1, 1].set_ylabel("Quantity")

output_path = os.path.join(augmentation_subdirectory_path, 'data_before_and_after_SMOTE.png')
plt.savefig(output_path)

plt.tight_layout()
plt.show()

# Normalize X values

min_max_scaler = preprocessing.MinMaxScaler()

final_x_train_aug = min_max_scaler.fit_transform(x_train_aug)
x_test = min_max_scaler.transform(x_test)

# Random Forest

print('Starting Random Forest...')

x_train_aug = x_train_aug.dropna(axis=0)
y_train_aug = y_train_aug.dropna(axis=0)

# Model Fit

rndforest = RandomForestClassifier(random_state=42)
rndforest.fit(x_train_aug, y_train_aug)

# Prediction
y_pred = rndforest.predict(x_test)

print(f'Y pred length: {len(y_pred)}')

# Evaluation
matriz_conf = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_conf, display_labels=rndforest.classes_)
disp.plot()
plt.show()

recall_score = recall_score(y_test, y_pred, average='weighted', zero_division=1)
print(f'Recall: {recall_score}')


# XGBOOST

# Label Encoding for XGBoost

label_encoder = LabelEncoder()

y_concatenated = pd.concat([y_train_aug, y_test])
y_concatenated_encoded = label_encoder.fit_transform(y_concatenated)

y_train_aug_encoded = y_concatenated_encoded[:len(y_train_aug)]
y_test_encoded = y_concatenated_encoded[len(y_train_aug):]

# Model Fit
print('Starting XGBoost...')

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(x_train_aug, y_train_aug_encoded)

# Predict
y_pred_encoded = xgb_model.predict(x_test)

# Decode the predictions back to the original class labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluation
matriz_conf = confusion_matrix(y_test, y_pred)
print(matriz_conf)

recall_score = recall_score(y_test, y_pred, average='weighted', zero_division=1)
print(f'Recall: {recall_score}')
