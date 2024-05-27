import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn import preprocessing
from datetime import datetime
from xgboost import XGBClassifier
from joblib import dump, load
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Make plots and prints?
make_plots = 0

# Import test and train data?
import_test_train_data = 0

# Load models?
load_models = 1

# Data Augmentation?
augmentation = 0
print(f"Data Augmentation: {augmentation}")

# Final data arrays
x_train = pd.DataFrame()
x_test = pd.DataFrame()
y_train = pd.Series(dtype='float64')
y_test = pd.Series(dtype='float64')

# Data paths
clean_train_data_path = r'../data/clean_data/cleaned_data2022_2023_2024.xlsx'
clean_test_data_path = r'../data/clean_data/cleaned_data_2022_line410.xlsx'

# Load split test and train data
if import_test_train_data == 1:

    print(f'\nLoading test and train data...')

    if augmentation == 1:

        x_train = pd.read_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_x_train_aug.xlsx')

        y_train = pd.read_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_y_train_aug.xlsx')['Group']

    else:

        x_train = pd.read_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_x_train.xlsx')

        y_train = pd.read_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_y_train.xlsx')['Group']

    x_test = pd.read_excel(r'..\data\split_train_test_data\defect_groups\defect_groups_x_test.xlsx')

    y_test = pd.read_excel(r'..\data\split_train_test_data\defect_groups\defect_groups_y_test.xlsx')[
        'Group']

    print(f'Loaded test and train data!')

# Configure logging file
logging.basicConfig(filename=f'logs/just_defects_prediction_model_metrics_log.txt', level=logging.INFO)
logging.info(f"\nTimestamp: {timestamp}")
file_name = os.path.splitext(os.path.basename(clean_train_data_path))[0]
logging.info(f"\nFile: {file_name}")

if import_test_train_data == 0:

    print(f'\nPreparing test and train data...')

    # Directory to save the resulting plots
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Load clean train data
    df_train = pd.read_excel(clean_train_data_path)

    # Load clean test data
    df_test = pd.read_excel(clean_test_data_path)

    # Remove samples that don't have defects
    df_train = df_train[df_train["Defect Code"] != 0]
    df_test = df_test[df_test["Defect Code"] != 0]

    # Removing the Recording Date and targets from the data
    y_train = df_train["Group"]
    y_test = df_test["Group"]

    x_train = df_train.drop(["Recording Date", "Defect Code", "Group"], axis=1)
    x_test = df_test.drop(["Recording Date", "Defect Code", "Group"], axis=1)

    print(f'Test and train data done!')

    if augmentation == 1:

        print(f'\nStarting data augmentation...')

        # Calculate the quantity of each defect group in the train set
        defect_count = y_train.value_counts()

        # Data Augmentation using SMOTE
        x_train, y_train = SMOTE(random_state=42).fit_resample(x_train, y_train)

        # Calculate the quantity of each defect group in the train set after augmentation
        defect_count_aug = y_train.value_counts()

        print(f'\nData augmentation done!')

        if make_plots == 1:
            aug_plots_dir = "../plots/augmentation"
            os.makedirs(plots_dir, exist_ok=True)

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

            # Number of occurrences for each class of defects before augmentation
            sns.barplot(x=defect_count.index, y=defect_count.values, palette="magma", ax=axes[0])
            axes[0].set_title("ORIGINAL - Quantity of the different defect codes in train data")
            axes[0].set_xlabel("Group")
            axes[0].set_ylabel("Quantity")

            # Number of occurrences for each class of defects after augmentation
            sns.barplot(x=defect_count_aug.index, y=defect_count_aug.values, palette="magma", ax=axes[1])
            axes[1].set_title("AUGMENTED - Quantity of the different defect codes in train data")
            axes[1].set_xlabel("Group")
            axes[1].set_ylabel("Quantity")

            output_path = os.path.join(aug_plots_dir, 'defect_groups_before_and_after.png')
            plt.savefig(output_path)

            plt.tight_layout()
            plt.show()

    else:

        x_train, y_train = x_train, y_train

    column_names = x_train.columns

    # Save test and train data
    print(f'\nSaving data...')

    if augmentation == 1:

        pd.DataFrame(x_train, columns=column_names).to_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_x_train_aug.xlsx',
            index=False)

        pd.Series(y_train, name='Group').to_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_y_train_aug.xlsx',
            index=False)

    else:

        pd.DataFrame(x_train, columns=column_names).to_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_x_train.xlsx',
            index=False)

        pd.Series(y_train, name='Group').to_excel(
            r'..\data\split_train_test_data\defect_groups\defect_groups_y_train.xlsx',
            index=False)

    pd.DataFrame(x_test, columns=column_names).to_excel(
        r'..\data\split_train_test_data\defect_groups\defect_groups_x_test.xlsx',
        index=False)

    pd.Series(y_test, name='Group').to_excel(
        r'..\data\split_train_test_data\defect_groups\defect_groups_y_test.xlsx',
        index=False)

    print(f'Saved data!')

print("-------------- Modelling --------------")

#################
# Random Forest #
#################

print(f'\nStarting Random Forest...')

# Model Fit

rndforest = RandomForestClassifier(random_state=42)

if load_models == 1:
    rndforest = load(r'models/defect_groups/defect_groups_random_forest_model.pkl')
else:
    rndforest.fit(x_train, y_train)

# Prediction
y_pred_rf = rndforest.predict(x_test)

# Evaluation
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf, display_labels=rndforest.classes_)
disp_rf.plot()
plt.title('Random Forest Confusion Matrix')
plt.savefig(r'..\plots\confusion_matrix\defect_groups\rf.png')
plt.show()

recall_score_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=1)
print(f'Recall: {recall_score_rf:.6f}')

precision_score_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=1)
print(f'Precision: {precision_score_rf:.6f}')

logging.info("\nRandomForest Metrics:")
logging.info(f"Recall: {recall_score_rf:.6f}")
logging.info(f"Precision: {precision_score_rf:.6f}")

# Save
if load_models == 0:
    dump(rndforest, r'models/defect_groups/defect_groups_random_forest_model.pkl')

###########
# XGBOOST #
###########

# Label Encoding for XGBoost
label_encoder = LabelEncoder()

y_concatenated = pd.concat([y_train, y_test])
y_concatenated_encoded = label_encoder.fit_transform(y_concatenated)

y_train_encoded = y_concatenated_encoded[:len(y_train)]
y_test_encoded = y_concatenated_encoded[len(y_train):]

# Model Fit
print(f'\nStarting XGBoost...')

xgb_model = XGBClassifier(random_state=42)

if load_models == 1:
    xgb_model.load_model(r'models\defect_groups\defect_groups_xgb_model.json')
else:
    xgb_model.fit(x_train, y_train_encoded)

# Predict
y_pred_encoded = xgb_model.predict(x_test)

# Decode the predictions back to the original class labels
y_pred_xgb = label_encoder.inverse_transform(y_pred_encoded)

# Evaluation

# Display the confusion matrix
confusion_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_xgb, display_labels=xgb_model.classes_)
disp_xgb.plot()
plt.title('XGBoost Confusion Matrix')
plt.savefig(r'..\plots\confusion_matrix\defect_groups\xgboost.png')
plt.show()

recall_score_xgb = recall_score(y_test, y_pred_xgb, average='weighted', zero_division=1)
print(f'Recall: {recall_score_xgb:.6f}')

precision_score_xgb = precision_score(y_test, y_pred_xgb, average='weighted', zero_division=1)
print(f'Precision: {precision_score_xgb:.6f}')

logging.info("XGBoost Metrics:")
logging.info(f"Recall: {recall_score_xgb:.6f}")
logging.info(f"Precision: {precision_score_xgb:.6f}")

# Save
if load_models == 0:
    xgb_model.save_model(r'models\defect_groups\defect_groups_xgb_model.json')

# #######
# # SVM #
# #######
#
# print(f'\nStarting SVM...')
#
# svm_model = SVC(random_state=42, probability=True, decision_function_shape='ovo')
#
# if load_models == 1:
#     svm_model = load(r'..\models\defect_groups\defect_groups_svm_model.pkl')
# else:
#     svm_model.fit(x_train_aug, y_train_aug)
#
# # Predict
# y_pred_svm = svm_model.predict(x_test)
#
# # Evaluation
#
# confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
# disp_svm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_svm, display_labels=svm_model.classes_)
# disp_svm.plot()
# plt.title('SVM Confusion Matrix')
# plt.savefig(r'..\plots\confusion_matrix\defect_groups\svm.png')
# plt.show()
#
# recall_score_svm = recall_score(y_test, y_pred_svm, average='weighted', zero_division=1)
# print(f'Recall: {recall_score_svm:.6f}')
#
# precision_score_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=1)
# print(f'Precision: {precision_score_svm:.6f}')
#
# logging.info("\nSVM Metrics:")
# logging.info(f"Recall: {recall_score_svm:.6f}")
# logging.info(f"Precision: {precision_score_svm:.6f}")
#
# # Save
# if load_models == 0:
#     dump(svm_model, r'..\models\defect_groups\defect_groups_svm_model.pkl')

############
# CATBOOST #
############

print(f'\nStarting CatBoost...')

catboost_model = CatBoostClassifier(loss_function='MultiClass', verbose=False)

if load_models == 1:
    catboost_model = CatBoostClassifier(loss_function='MultiClass', verbose=False)
    catboost_model.load_model(r'models\defect_groups\defect_groups_catboost_model.cbm')
else:
    catboost_model.fit(x_train, y_train)

# Predict
y_pred_cat = catboost_model.predict(x_test)

# Evaluation
confusion_matrix_cat = confusion_matrix(y_test, y_pred_cat)
disp_cat = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cat, display_labels=catboost_model.classes_)
disp_cat.plot()
plt.title('Catboost Confusion Matrix')
plt.savefig(r'..\plots\confusion_matrix\defect_groups\catboost.png')
plt.show()

recall_score_cat = recall_score(y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Recall: {recall_score_cat:.6f}')

precision_score_cat = precision_score(y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Precision: {precision_score_cat:.6f}')

logging.info("\nCatBoost Metrics:")
logging.info(f"Recall: {recall_score_cat:.6f}")
logging.info(f"Precision: {precision_score_cat:.6f}")

# Save
if load_models == 0:
    catboost_model.save_model(r'models\defect_groups\defect_groups_catboost_model.cbm')

#####################
# ENSEMBLE / VOTING #
#####################

print(f'\nStarting Ensemble...')

# estimators = [
#     ('random_forest', rndforest),
#     ('xgboost', xgb_model),
#     ('catboost', catboost_model)
# ]
#
# voting_classifier = VotingClassifier(estimators, voting='hard')
#
# if load_models == 1:
#     voting_classifier = load(r'models\defect_groups\defect_groups_voting_model.pkl')
#
# else:
#     voting_classifier.fit(x_train, y_train)
#
# x_test_reshaped = x_test.reshape(-1)
# y_pred_ensemble = voting_classifier.predict(x_test_reshaped)

class CustomVotingClassifier(VotingClassifier):
    def _predict(self, X):
        predictions = [est.predict(X).reshape(-1) for est in self.estimators_]
        return np.asarray(predictions).T

# Define your estimators
estimators = [
    ('random_forest', rndforest),
    ('xgboost', xgb_model),
    ('catboost', catboost_model)
]

# Use the custom voting classifier
voting_classifier = CustomVotingClassifier(estimators, voting='hard', flatten_transform=True)

# Load or train the model
if load_models == 1:
    voting_classifier = load(r'models\just_defects\just_defects_voting_model.pkl')
else:
    voting_classifier.fit(x_train, y_train)

# Make predictions
y_pred_ensemble = voting_classifier.predict(x_test)

# To ensure all labels are strings
# y_test = y_test.astype(str)
# y_pred_ensemble = y_pred_ensemble.astype(str)

# # Evaluation
# confusion_matrix_en = confusion_matrix(y_test, y_pred_ensemble)
# disp_en = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_en, display_labels=voting_classifier.classes_)
# disp_en.plot()
# plt.title('Ensemble Model Confusion Matrix')
# plt.savefig(os.path.join(r'..\plots\confusion_matrix\defect_groups', 'ensemble.png'))
# plt.show()

recall_score_en = recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=1)
print(f'Recall: {recall_score_en:.6f}')

precision_score_en = precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=1)
print(f'Precision: {precision_score_en:.6f}')

logging.info("\nEnsemble Metrics:")
logging.info(f"Recall: {recall_score_en:.6f}")
logging.info(f"Precision: {precision_score_en:.6f}")

# Save
if load_models == 0:
    dump(voting_classifier, r'models\defect_groups\defect_groups_voting_model.pkl')
