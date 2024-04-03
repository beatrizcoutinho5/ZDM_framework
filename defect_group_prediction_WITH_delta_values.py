import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn import preprocessing
from xgboost import XGBClassifier
from joblib import dump, load
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Make plots and prints?
make_plots = 0

# Import test and train data?
import_test_train_data = 1

# Load models?
load_models = 1

# Perfom Grid Search?
grid_search = 0

# Final data arrays
final_x_train = pd.DataFrame()
final_x_test = pd.DataFrame()
final_y_train = pd.Series(dtype='float64')
final_y_test = pd.Series(dtype='float64')

clean_data_delta_path = r'data\clean_data\cleaned_data_with_deltavalues2022_2023_2024.xlsx'
test_data_path = r'data\clean_data\cleaned_data_with_deltavalues_2022_line410.xlsx'

if import_test_train_data == 1:

    print(f'\nLoading test and train data...')

    x_train_aug = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\x_train_aug.xlsx')

    y_train_aug = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\y_train_aug.xlsx')[
        'Defect Code']

    x_test = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\x_test.xlsx')

    final_y_test = pd.read_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\y_test.xlsx')[
        'Defect Code']

    print(f'Loaded test and train data!')


# Configure logging file
# for metrics and grid search
logging.basicConfig(filename=f'prediction_model_metrics_log.txt', level=logging.INFO)
logging.info(f"\nTimestamp: {timestamp}")
file_name = os.path.splitext(os.path.basename(clean_data_delta_path))[0]
logging.info(f"\nFile: {file_name}")

logging.info(f"\n ---------- TESTE COM GRUPOS DE DEFEITOS ----------")

# removing samples without defect
df = pd.read_excel(clean_data_delta_path)
df = df[df["Defect Code"] != 0]

df_test = pd.read_excel(test_data_path)
df_test = df_test[df_test["Defect Code"] != 0]

if import_test_train_data == 0:

    final_y_train = df["Group"]
    final_x_train = df.drop("Group", axis=1)

    final_y_test = df_test["Group"]
    final_x_test = df_test.drop("Group", axis=1)

    # Removing the date from the data
    final_x_train = final_x_train.drop("Recording Date", axis=1)
    final_x_test = final_x_test.drop("Recording Date", axis=1)

    # Data Augmentation using SMOTE
    x_train_aug, y_train_aug = SMOTE(random_state=42).fit_resample(final_x_train, final_y_train)

    # Normalize X values
    min_max_scaler = preprocessing.MinMaxScaler()

    x_train_aug = min_max_scaler.fit_transform(x_train_aug)
    x_test = min_max_scaler.transform(final_x_test)

    # Save test and train data
    print(f'\nSaving test and train data...')

    pd.DataFrame(x_train_aug, columns=final_x_train.columns).to_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\x_train_aug.xlsx',
        index=False)

    pd.Series(y_train_aug, name='Defect Code').to_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\y_train_aug.xlsx',
        index=False)

    pd.DataFrame(x_test, columns=final_x_test.columns).to_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\x_test.xlsx',
        index=False)

    pd.Series(final_y_test, name='Defect Code').to_excel(
        r'data\split_train_test_data\with_delta_values\defect_groups\y_test.xlsx',
        index=False)

    print(f'\nSaved train and test data!')

print(f'\n-------------- ML MODELS --------------')


def multiclass_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')


scorer = make_scorer(multiclass_recall_score)

#################
# Random Forest #
#################

print(f'\nStarting Random Forest...')

# Model Fit

rndforest = RandomForestClassifier(random_state=42, max_depth=20, min_samples_split=5, n_estimators=200)

if grid_search == 1:
    param_grid_rf = {
        'n_estimators': [200, 300],
        'max_depth': [10, 20],
        'min_samples_split': [5, 10],
    }

    grid_search_rf = GridSearchCV(rndforest, param_grid_rf, cv=5, scoring=scorer, verbose=4)
    grid_search_rf.fit(x_train_aug, y_train_aug)

    best_params_rf = grid_search_rf.best_params_
    best_recall_rf = grid_search_rf.best_score_

    print("Best parameters for Random Forest:", best_params_rf)
    print("Best recall for Random Forest:", best_recall_rf)

    logging.info("\nRandomForest GRID SEARCH:")
    logging.info(f"Best parameters: {best_params_rf}")

if load_models == 1:
    rndforest = load(r'models\with_delta_values\defect_groups\random_forest_model.pkl')
else:
    rndforest.fit(x_train_aug, y_train_aug)

# Prediction
y_pred_rf = rndforest.predict(x_test)

# Evaluation

confusion_matrix_rf = confusion_matrix(final_y_test, y_pred_rf)
plt.figure(figsize=(8, 16))
disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf, display_labels=rndforest.classes_)
disp_rf.plot()
plt.xticks(rotation=60)
plt.title('RF Confusion Matrix - with "Defect Class"')
plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.4)
plt.savefig(r'plots\confusion_matrix\with_delta_values\defect_groups\rf.png')
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
    dump(rndforest, r'models\with_delta_values\defect_groups\random_forest_model.pkl')

###########
# XGBOOST #
###########

# Label Encoding for XGBoost

label_encoder = LabelEncoder()

y_concatenated = pd.concat([y_train_aug, final_y_test])
y_concatenated_encoded = label_encoder.fit_transform(y_concatenated)

y_train_aug_encoded = y_concatenated_encoded[:len(y_train_aug)]
y_test_encoded = y_concatenated_encoded[len(y_train_aug):]

original_labels = label_encoder.classes_

# Model Fit

print(f'\nStarting XGBoost...')

xgb_model = XGBClassifier(random_state=42)


if grid_search == 1:
    param_grid_xgb = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [6, 9],
        'min_child_weight': [1, 3],
        'gamma': [0.01, 0.2],
    }

    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring=scorer, verbose=4)
    grid_search_xgb.fit(x_train_aug, y_train_aug_encoded)
    best_params_xgb = grid_search_xgb.best_params_
    best_recall_xgb = grid_search_xgb.best_score_
    print("Best parameters for XGBoost:", best_params_xgb)
    print("Best recall for XGBoost:", best_recall_xgb)

    logging.info("\nXGBoost GRID SEARCH:")
    logging.info(f"Best parameters: {best_params_xgb}")

if load_models == 1:
    xgb_model.load_model(r'models\with_delta_values\defect_groups\xgb_model.json')
else:
    xgb_model.fit(x_train_aug, y_train_aug_encoded)

# Predict
y_pred_encoded = xgb_model.predict(x_test)

# Decode the predictions back to the original class labels
y_pred_xgb = label_encoder.inverse_transform(y_pred_encoded)

# Evaluation

# Display the confusion matrix
confusion_matrix_xgb = confusion_matrix(final_y_test, y_pred_xgb)
plt.figure(figsize=(8, 16))
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_xgb, display_labels=original_labels)
disp_xgb.plot()
plt.xticks(rotation=60)
plt.title('XGBoost Confusion Matrix')
plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.4)
plt.savefig(r'plots\confusion_matrix\with_delta_values\defect_groups\xgboost.png')
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
    xgb_model.save_model(r'models\with_delta_values\defect_groups\xgb_model.json')

############
# CATBOOST #
############

print(f'\nStarting CatBoost...')

catboost_model = CatBoostClassifier(loss_function='MultiClass', verbose=False)

if grid_search == 1:
    param_grid_cat = {
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6],
        'l2_leaf_reg': [1, 3]
    }

    grid_search_cat = catboost_model.grid_search(param_grid_cat, X=Pool(x_train_aug, label=y_train_aug), cv=5,
                                                 refit=True, partition_random_seed=42, verbose=4)
    best_params_cat = grid_search_cat['params']
    best_recall_cat = grid_search_cat['cv_results']['test-MultiClassRecall-mean'].max()
    print("Best parameters for CatBoost:", best_params_cat)
    print("Best recall for CatBoost:", best_recall_cat)

    logging.info("\nCatBoost GRID SEARCH:")
    logging.info(f"Best parameters: {best_params_cat}")

if load_models == 1:
    catboost_model = CatBoostClassifier(loss_function='MultiClass', verbose=False)
    catboost_model.load_model(r'models\with_delta_values\defect_groups\catboost_model.cbm')
else:
    catboost_model.fit(x_train_aug, y_train_aug)

# Predict
y_pred_cat = catboost_model.predict(x_test)

# Evaluation

confusion_matrix_cat = confusion_matrix(final_y_test, y_pred_cat)
plt.figure(figsize=(8, 16))
disp_cat = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cat, display_labels=catboost_model.classes_)
disp_cat.plot()
plt.xticks(rotation=60)
plt.title('CATBOOST Confusion Matrix')
plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.4)
plt.savefig(r'plots\confusion_matrix\with_delta_values\defect_groups\catboost.png')
plt.show()

recall_score_cat = recall_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Recall: {recall_score_cat:.6f}')

precision_score_cat = precision_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
print(f'Precision: {precision_score_cat:.6f}')

logging.info("\nCatBoost Metrics:")
logging.info(f"Recall: {recall_score_cat:.6f}")
logging.info(f"Precision: {precision_score_cat:.6f}")

# # Save
if load_models == 0:
    catboost_model.save_model(r'models\with_delta_values\defect_groups\catboost_model.cbm')
