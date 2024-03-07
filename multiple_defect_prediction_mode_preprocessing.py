import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

clean_data_delta_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\cleaned_data_with_deltavalues2022_2023_2024.xlsx'

df = pd.read_excel(clean_data_delta_path)

# Make plots and prints?
make_plots = 0

# Final data arrays

final_x_train = pd.DataFrame()
final_x_test = pd.DataFrame()
final_y_train = pd.Series(dtype='float64')
final_y_test = pd.Series(dtype='float64')

# Directory to save the resulting plots
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# # Removing the date from the data
#
# df = df.drop("Recording Date", axis=1)

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

# Split the df by year

df["Recording Date"] = pd.to_datetime(df["Recording Date"], format="%d/%m/%Y %H:%M:%S")

# Separate the DataFrame into different DataFrames based on years
years = df["Recording Date"].dt.year.unique()

dfs_by_year = {}
for year in years:
    dfs_by_year[year] = df[df["Recording Date"].dt.year == year]

# Split train / test

for year in years:
    dfs_by_year[year] = df[df["Recording Date"].dt.year == year]

    split_index = int(0.8 * len(dfs_by_year[year]))

    train_data = dfs_by_year[year].iloc[:split_index]
    test_data = dfs_by_year[year].iloc[split_index:]

    y_train = train_data["Defect Code"]
    x_train = train_data.drop("Defect Code", axis=1)

    y_test = test_data["Defect Code"]
    x_test = test_data.drop("Defect Code", axis=1)

    final_x_train = pd.concat([final_x_train, x_train])
    final_y_train = pd.concat([final_y_train, y_train])

    final_x_test = pd.concat([final_x_test, x_test])
    final_y_test = pd.concat([final_y_test, y_test])

# Removing the date from the data

final_x_train = final_x_train.drop("Recording Date", axis=1)
final_x_test = final_x_test.drop("Recording Date", axis=1)

# Calculate the quantity of each defect code in the train set
defect_count = final_y_train.value_counts()
defect_count_without_zero = final_y_train[final_y_train != 0].value_counts()

# Data Augmentation using SMOTE

x_train_aug, y_train_aug = SMOTE(random_state=42).fit_resample(final_x_train, final_y_train)

defect_count_aug = y_train_aug.value_counts()
defect_count_without_zero_aug = y_train_aug[y_train_aug != 0].value_counts()

# Plot the frequency of defects before and after data augmentation
if make_plots == 1:
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

x_train_aug = min_max_scaler.fit_transform(x_train_aug)
x_test = min_max_scaler.transform(final_x_test)

#################
# Random Forest #
#################

print(f'\nStarting Random Forest...')

# Model Fit

rndforest = RandomForestClassifier(random_state=42)
rndforest.fit(x_train_aug, y_train_aug)

# Prediction
y_pred_rf = rndforest.predict(x_test)

# Evaluation

confusion_matrix_rf = confusion_matrix(final_y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf, display_labels=rndforest.classes_)
disp_rf.plot()
plt.title('RF Confusion Matrix - with "Defect Class"')
plt.show()

# Display the confusion matrix without class '0' -> no defect
labels_rf = np.unique(final_y_test)[1:]
confusion_matrix_xgb = confusion_matrix(final_y_test, y_pred_rf, labels=labels_rf)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_xgb, display_labels=labels_rf)
disp_xgb.plot()
plt.title('RF Confusion Matrix - without "Defect Class"')
plt.show()

recall_score_rf = recall_score(final_y_test, y_pred_rf, average='weighted', zero_division=1)
print(f'Recall: {recall_score_rf:.6f}')

precision_score_rf = precision_score(final_y_test, y_pred_rf, average='weighted', zero_division=1)
print(f'Precision: {precision_score_rf:.6f}')

###########
# XGBOOST #
###########

# Label Encoding for XGBoost

label_encoder = LabelEncoder()

y_concatenated = pd.concat([y_train_aug, final_y_test])
y_concatenated_encoded = label_encoder.fit_transform(y_concatenated)

y_train_aug_encoded = y_concatenated_encoded[:len(y_train_aug)]
y_test_encoded = y_concatenated_encoded[len(y_train_aug):]

# Model Fit
print(f'\nStarting XGBoost...')

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(x_train_aug, y_train_aug_encoded)

# Predict
y_pred_encoded = xgb_model.predict(x_test)

# Decode the predictions back to the original class labels
y_pred_xgb = label_encoder.inverse_transform(y_pred_encoded)

# Evaluation

# Display the confusion matrix
confusion_matrix_xgb = confusion_matrix(final_y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_xgb, display_labels=xgb_model.classes_)
disp_xgb.plot()
plt.title('XGBoost Confusion Matrix')
plt.show()

# Display the confusion matrix without class '0' -> no defect
labels_xgb = np.unique(final_y_test)[1:]
confusion_matrix_xgb = confusion_matrix(final_y_test, y_pred_xgb, labels=labels_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_xgb, display_labels=labels_xgb)
disp_xgb.plot()
plt.title('XGBoost Confusion Matrix - without "Defect Class"')
plt.show()

recall_score_xgb = recall_score(final_y_test, y_pred_xgb, average='weighted', zero_division=1)
print(f'Recall: {recall_score_xgb:.6f}')

precision_score_xgb = precision_score(final_y_test, y_pred_xgb, average='weighted', zero_division=1)
print(f'Precision: {precision_score_xgb:.6f}')

#######
# SVM #
#######

print(f'\nStarting SVM...')

svm_model = SVC(random_state=42)
svm_model.fit(x_train_aug, y_train_aug)

# Predict
y_pred_svm = svm_model.predict(x_test)

# Evaluation

confusion_matrix_svm = confusion_matrix(final_y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_svm, display_labels=svm_model.classes_)
disp_svm.plot()
plt.title('SVM Confusion Matrix')
plt.show()

recall_score_svm = recall_score(final_y_test, y_pred_svm, average='weighted', zero_division=1)
print(f'Recall: {recall_score_svm:.6f}')

precision_score_svm = precision_score(final_y_test, y_pred_svm, average='weighted', zero_division=1)
print(f'Precision: {precision_score_svm:.6f}')

############
# CATBOOST #
############

print(f'\nStarting CatBoost...')

catboost_model = CatBoostClassifier(loss_function='MultiClass', verbose=False)
catboost_model.fit(x_train_aug, y_train_aug)

# Predict
y_pred_cat = catboost_model.predict(x_test)

# Evaluation
recall_score_cat = recall_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Recall: {recall_score_cat:.6f}')

precision_score_cat = precision_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Precision: {precision_score_cat:.6f}')
