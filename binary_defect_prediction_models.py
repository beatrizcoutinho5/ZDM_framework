import os
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from datetime import datetime
from sklearn import preprocessing
from xgboost import XGBClassifier
from joblib import dump, load
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Make plots and prints?
make_plots = 1

# Import test and train data?
import_test_train_data = 0

# Load models?
# 0 -> train the models
# 1 -> import trained models for metrics
load_models = 0

# Final data arrays

final_x_train = pd.DataFrame()
final_x_test = pd.DataFrame()
final_y_train = pd.Series(dtype='float64')
final_y_test = pd.Series(dtype='float64')


clean_data_delta_paths = [
    r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\cleaned_data_with_deltavalues2022_2023_2024.xlsx']


if import_test_train_data == 1:
    print(f'\nLoading test and train data...')

    x_train_aug = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_x_train_aug.xlsx')

    y_train_aug = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_y_train_aug.xlsx')['Defect Code']

    x_test = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_x_test.xlsx')

    final_y_test = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_y_test.xlsx')['Defect Code']

    print(f'Loaded test and train data!')


for clean_data_delta_path in clean_data_delta_paths:

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
        df['Binary Defect Code'] = np.where(df['Defect Code'] == 0, 0, 1)

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

            y_train = train_data["Binary Defect Code"]
            x_train = train_data.drop("Binary Defect Code", axis=1)

            y_test = test_data["Binary Defect Code"]
            x_test = test_data.drop("Binary Defect Code", axis=1)

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

        x_train_aug, y_train_aug = final_x_train, final_y_train
        y_train_aug = final_y_train

        # Normalize X values

        min_max_scaler = preprocessing.MinMaxScaler()

        x_train_aug = min_max_scaler.fit_transform(x_train_aug)
        x_test = min_max_scaler.transform(final_x_test)

        # Save test and train data
        print(f'\nSaving test and train data...')

        pd.DataFrame(x_train_aug, columns=final_x_train.columns).to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_x_train_aug.xlsx',
            index=False)

        pd.Series(y_train_aug, name='Defect Code').to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_y_train_aug.xlsx',
            index=False)

        pd.DataFrame(x_test, columns=final_x_test.columns).to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_x_test.xlsx',
            index=False)

        pd.Series(final_y_test, name='Defect Code').to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_y_test.xlsx',
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
        rndforest = load(r'models\with_delta_values\binary_random_forest_model.pkl')
    else:
        rndforest.fit(x_train_aug, y_train_aug)

    # Prediction
    y_pred_rf = rndforest.predict(x_test)

    # Evaluation

    confusion_matrix_rf = confusion_matrix(final_y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf, display_labels=rndforest.classes_)
    disp_rf.plot()
    plt.title('RF Confusion Matrix - with "Defect Class"')
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
        dump(rndforest, r'models\with_delta_values\binary_random_forest_model.pkl')

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
        xgb_model.load_model(r'models\with_delta_values\binary_xgb_model.model')
    else:
        xgb_model.fit(x_train_aug, y_train_aug_encoded)

    # Predict
    y_pred_encoded = xgb_model.predict(x_test)

    # Decode the predictions back to the original class labels
    y_pred_xgb = label_encoder.inverse_transform(y_pred_encoded)

    # Evaluation

    recall_score_xgb = recall_score(final_y_test, y_pred_xgb, average='weighted', zero_division=1)
    print(f'Recall: {recall_score_xgb:.6f}')

    precision_score_xgb = precision_score(final_y_test, y_pred_xgb, average='weighted', zero_division=1)
    print(f'Precision: {precision_score_xgb:.6f}')

    logging.info("XGBoost Metrics:")
    logging.info(f"Recall: {recall_score_xgb:.6f}")
    logging.info(f"Precision: {precision_score_xgb:.6f}")

    # Save
    if load_models == 0:
        xgb_model.save_model('models/with_delta_values/binary_xgb_model.json')

    #######
    # SVM #
    #######

    print(f'\nStarting SVM...')

    svm_model = SVC(random_state=42, probability=True)

    if load_models == 1:
        svm_model = load(r'models\with_delta_values\binary_svm_model.pkl')
    else:
        svm_model.fit(x_train_aug, y_train_aug)

    # Predict
    y_pred_svm = svm_model.predict(x_test)

    # Evaluation

    recall_score_svm = recall_score(final_y_test, y_pred_svm, average='weighted', zero_division=1)
    print(f'Recall: {recall_score_svm:.6f}')

    precision_score_svm = precision_score(final_y_test, y_pred_svm, average='weighted', zero_division=1)
    print(f'Precision: {precision_score_svm:.6f}')

    logging.info("\nSVM Metrics:")
    logging.info(f"Recall: {recall_score_svm:.6f}")
    logging.info(f"Precision: {precision_score_svm:.6f}")

    # Save
    if load_models == 0:
        dump(svm_model, r'models\with_delta_values\binary_svm_model.pkl')


    ############
    # CATBOOST #
    ############

    print(f'\nStarting CatBoost...')

    catboost_model = CatBoostClassifier(loss_function='Logloss', verbose=False)

    if load_models == 1:
        catboost_model = CatBoostClassifier(loss_function='MultiClass', verbose=False)
        catboost_model.load_model(r'models\with_delta_values\binary_catboost_model.cbm')
    else:
        catboost_model.fit(x_train_aug, y_train_aug)

    # Predict
    y_pred_cat = catboost_model.predict(x_test)

    # Evaluation
    recall_score_cat = recall_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
    print(f'Recall: {recall_score_cat:.6f}')

    precision_score_cat = precision_score(final_y_test, y_pred_cat, average='weighted', zero_division=1)
    print(f'Precision: {precision_score_cat:.6f}')

    logging.info("\nCatBoost Metrics:")
    logging.info(f"Recall: {recall_score_cat:.6f}")
    logging.info(f"Precision: {precision_score_cat:.6f}")

    # Save
    if load_models == 0:
        catboost_model.save_model(r'models\with_delta_values\binary_catboost_model.cbm')
