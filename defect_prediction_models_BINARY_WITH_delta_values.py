import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from datetime import datetime
from xgboost import XGBClassifier
from joblib import dump, load
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings("ignore", category=UserWarning)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Make plots and prints?
make_plots = 0

# Import test and train data?
import_test_train_data = 1

# Load models?
# 0 -> train the models
# 1 -> import trained models for metrics
load_models = 0

# Final data arrays

final_x_train = pd.DataFrame()
final_x_test = pd.DataFrame()
final_y_train = pd.Series(dtype='float64')
final_y_test = pd.Series(dtype='float64')

clean_data_delta_path = r'data\clean_data\binary_cleaned_data_with_deltavalues_2022_2023_2024.xlsx'

if import_test_train_data == 1:
    print(f'\nLoading test and train data...')

    x_train_aug = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_x_train_aug.xlsx')

    y_train_aug = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_y_train_aug.xlsx')[
        'Defect Code']

    x_test = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_x_test.xlsx')

    final_y_test = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_y_test.xlsx')[
        'Defect Code']

    print(f'Loaded test and train data!')



# Configure logging file
logging.basicConfig(filename=f'binary_prediction_model_metrics_log.txt', level=logging.INFO)
logging.info(f"\nTimestamp: {timestamp}")
file_name = os.path.splitext(os.path.basename(clean_data_delta_path))[0]
logging.info(f"\nFile: {file_name}")

df = pd.read_excel(clean_data_delta_path)

if import_test_train_data == 0:

    # Directory to save the resulting plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # For binary classification
    # new column where '0' is one class (no defect) and any other code is the second class (defect)
    # df['Binary Defect Code'] = np.where(df['Defect Code'] == 0, 0, 1)

    df_line410_2022 = pd.read_excel(r'data\clean_data\binary_cleaned_data_with_deltavalues_2022_line410.xlsx')
    # df_line410_2022['Binary Defect Code'] = np.where(df_line410_2022['Defect Code'] == 0, 0, 1)

    # Removing the date from the data

    # final_x_train = df.drop(["Recording Date", "Defect Code", "Binary Defect Code", "Group"], axis=1)
    # final_x_test = df_line410_2022.drop(["Recording Date", "Defect Code", "Binary Defect Code", "Group"], axis=1)

    final_y_train = df["Defect Code"]
    final_y_test = df_line410_2022["Defect Code"]

    final_x_train = df.drop(["Recording Date", "Defect Code", "Group"], axis=1)
    final_x_test = df_line410_2022.drop(["Recording Date", "Defect Code", "Group"], axis=1)

    # final_y_train = df["Binary Defect Code"]
    # final_y_test = df_line410_2022["Binary Defect Code"]



    # Calculate the quantity of each defect code in the train set
    defect_count = final_y_train.value_counts()
    defect_count_without_zero = final_y_train[final_y_train != 0].value_counts()

    # Data Augmentation using SMOTE
    smote = SMOTE(random_state=42)
    x_train_aug, y_train_aug = smote.fit_resample(final_x_train, final_y_train)

    # x_train_aug = final_x_train # comentar isto qnd fizer augmentation
    # y_train_aug = final_y_train

    x_test = final_x_test  # comentar esta qnd fizer a norm


    column_names = x_train_aug.columns

    # # Normalize X values
    # min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler.fit(x_train_aug)
    #
    # x_train_aug = min_max_scaler.transform(x_train_aug)
    # x_test = min_max_scaler.transform(final_x_test)
    #
    # x_train_aug = pd.DataFrame(x_train_aug, columns=column_names)
    # x_test = pd.DataFrame(x_test, columns=column_names)
    #
    # # Save scaling parameters
    #
    # scaling_params = pd.DataFrame({
    #     'feature': x_train_aug.columns,
    #     'min': min_max_scaler.data_min_,
    #     'scale': min_max_scaler.scale_
    #
    # })
    #
    # scaling_params.to_excel(
    #     r'data\split_train_test_data\with_delta_values\binary_data\binary_scaling_param.xlsx',
    #     index=False)

    # Save test and train data
    print(f'\nSaving test and train data...')

    pd.DataFrame(x_train_aug, columns=column_names).to_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_x_train_aug.xlsx',
        index=False)

    pd.Series(y_train_aug, name='Defect Code').to_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_y_train_aug.xlsx',
        index=False)

    pd.DataFrame(x_test, columns=column_names).to_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_x_test.xlsx',
        index=False)

    pd.Series(final_y_test, name='Defect Code').to_excel(
        r'data\split_train_test_data\with_delta_values\binary_data\binary_y_test.xlsx',
        index=False)

    print(f'\nSaved train and test data!')

print("-------------- Modelling --------------")

#################
# Random Forest #
#################

print(f'\nStarting Random Forest...')

# Model Fit

rndforest = RandomForestClassifier(random_state=42)

if load_models == 1:
    rndforest = load(r'models\with_delta_values\binary\binary_random_forest_model.pkl')
else:
    rndforest.fit(x_train_aug, y_train_aug)

# Prediction
y_pred_rf = rndforest.predict(x_test)

# Evaluation

confusion_matrix_rf = confusion_matrix(final_y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf, display_labels=rndforest.classes_)
disp_rf.plot()
plt.title('RF Confusion Matrix - with "Defect Class"')
plt.savefig(r'plots\confusion_matrix\with_delta_values\binary\rf.png')
plt.show()

recall_score_rf = recall_score(final_y_test, y_pred_rf, average='weighted', zero_division=1)
print(f'Recall: {recall_score_rf:.6f}')

precision_score_rf = precision_score(final_y_test, y_pred_rf, average='weighted', zero_division=1)
print(f'Precision: {precision_score_rf:.6f}')

logging.info("\nRandomForest Metrics:")
logging.info(f"Recall: {recall_score_rf:.6f}")
logging.info(f"Precision: {precision_score_rf:.6f}")

# Save
if load_models == 0:
    dump(rndforest, r'models\with_delta_values\binary\binary_random_forest_model.pkl')

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

if load_models == 1:
    xgb_model.load_model(r'models\with_delta_values\binary\binary_xgb_model.json')
else:
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
plt.savefig(r'plots\confusion_matrix\with_delta_values\binary\xgboost.png')
plt.show()

recall_score_xgb = recall_score(final_y_test, y_pred_xgb, average='weighted', zero_division=1)
print(f'Recall: {recall_score_xgb:.6f}')

precision_score_xgb = precision_score(final_y_test, y_pred_xgb, average='weighted', zero_division=1)
print(f'Precision: {precision_score_xgb:.6f}')

logging.info("XGBoost Metrics:")
logging.info(f"Recall: {recall_score_xgb:.6f}")
logging.info(f"Precision: {precision_score_xgb:.6f}")

# Save
if load_models == 0:
    xgb_model.save_model(r'models\with_delta_values\binary\binary_xgb_model.json')

############
# CATBOOST #
############

print(f'\nStarting CatBoost...')

catboost_model = CatBoostClassifier(loss_function='Logloss', verbose=False)

if load_models == 1:
    catboost_model = CatBoostClassifier(loss_function='Logloss', verbose=False)
    catboost_model.load_model(r'models\with_delta_values\binary\binary_catboost_model.cbm')
else:
    catboost_model.fit(x_train_aug, y_train_aug)

# Predict
y_pred_cat = catboost_model.predict(x_test)

# Evaluation
confusion_matrix_cat = confusion_matrix(final_y_test, y_pred_cat)
disp_cat = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cat, display_labels=catboost_model.classes_)
disp_cat.plot()
plt.title('CATBOOST Confusion Matrix')
plt.savefig(r'plots\confusion_matrix\with_delta_values\binary\catboost.png')
plt.show()

recall_score_cat = recall_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Recall: {recall_score_cat:.6f}')

precision_score_cat = precision_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Precision: {precision_score_cat:.6f}')

logging.info("\nCatBoost Metrics:")
logging.info(f"Recall: {recall_score_cat:.6f}")
logging.info(f"Precision: {precision_score_cat:.6f}")

# Save
if load_models == 0:
    catboost_model.save_model(r'models\with_delta_values\binary\binary_catboost_model.cbm')

#####################
# ENSEMBLE / VOTING #
#####################

print(f'\nStarting Ensemble...')

estimators = [
    ('random_forest', rndforest),
    ('xgboost', xgb_model),
    ('catboost', catboost_model)
]

voting_classifier = VotingClassifier(estimators, voting='hard')

if load_models == 1:
    voting_classifier = load(r'models\with_delta_values\binary\binary_voting_model.pkl')

else:
    voting_classifier.fit(x_train_aug, y_train_aug)

y_pred_ensemble = voting_classifier.predict(x_test)

# Evaluation
confusion_matrix_en = confusion_matrix(final_y_test, y_pred_ensemble)
disp_en = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_en, display_labels=voting_classifier.classes_)
disp_en.plot()
plt.title('ENSEMBLE Confusion Matrix')
plt.savefig(r'plots\confusion_matrix\with_delta_values\binary\ensemble.png')
plt.show()

recall_score_en = recall_score(final_y_test, y_pred_ensemble, average='weighted', zero_division=1)
print(f'Recall: {recall_score_en:.6f}')

precision_score_en = precision_score(final_y_test, y_pred_ensemble, average='weighted', zero_division=1)
print(f'Precision: {precision_score_en:.6f}')

logging.info("\nEnsemble Metrics:")
logging.info(f"Recall: {recall_score_en:.6f}")
logging.info(f"Precision: {precision_score_en:.6f}")

# Save
if load_models == 0:
    dump(voting_classifier, r'models\with_delta_values\binary\binary_voting_model.pkl')