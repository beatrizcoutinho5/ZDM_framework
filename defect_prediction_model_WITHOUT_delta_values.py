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
from catboost import CatBoostClassifier, Pool, cv
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Make plots and prints?
make_plots = 1

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


clean_data_paths = [
    r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\cleaned_data2022_2023_2024.xlsx']

if import_test_train_data == 1:
    print(f'\nLoading test and train data...')

    x_train_aug = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\x_train_aug.xlsx')

    y_train_aug = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\y_train_aug.xlsx')['Defect Code']

    x_test = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\x_test.xlsx')

    final_y_test = pd.read_excel(
        r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\y_test.xlsx')['Defect Code']

    print(f'Loaded test and train data!')


for clean_data_path in clean_data_paths:

    # Configure logging file
    logging.basicConfig(filename=f'prediction_model_metrics_log.txt', level=logging.INFO)
    logging.info(f"\nTimestamp: {timestamp}")
    file_name = os.path.splitext(os.path.basename(clean_data_path))[0]
    logging.info(f"\nFile: {file_name}")

    df = pd.read_excel(clean_data_path)

    if import_test_train_data == 0:

        # Directory to save the resulting plots
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

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
            sns.barplot(x=defect_count_without_zero.index, y=defect_count_without_zero.values, palette="viridis",
                        ax=axes[0, 1])
            axes[0, 1].set_title("Original - Quantity of the selected defect codes in TRAIN set (excluding code 0)")
            axes[0, 1].set_xlabel("Defect Code")
            axes[0, 1].set_ylabel("Quantity")

            # Plot the bar plot with Defect Code = 0 in the augmented data
            sns.barplot(x=defect_count_aug.index, y=defect_count_aug.values, palette="magma", ax=axes[1, 0])
            axes[1, 0].set_title("AUG - Quantity of the selected defect codes in TRAIN set + No Defect")
            axes[1, 0].set_xlabel("Defect Code")
            axes[1, 0].set_ylabel("Quantity")

            # Plot the bar plot with only defects (excluding Defect Code = 0) in the augmented data
            sns.barplot(x=defect_count_without_zero_aug.index, y=defect_count_without_zero_aug.values,
                        palette="viridis",
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

        # Save test and train data
        print(f'\nSaving test and train data...')

        pd.DataFrame(x_train_aug, columns=final_x_train.columns).to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\x_train_aug.xlsx',
            index=False)

        pd.Series(y_train_aug, name='Defect Code').to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\y_train_aug.xlsx',
            index=False)

        pd.DataFrame(x_test, columns=final_x_test.columns).to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\x_test.xlsx',
            index=False)

        pd.Series(final_y_test, name='Defect Code').to_excel(
            r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\without_delta_values\y_test.xlsx',
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

    rndforest = RandomForestClassifier(random_state=42, max_depth= 20, min_samples_split= 5, n_estimators= 200)

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

    if load_models == 1:
        rndforest = load(r'models\without_delta_values\random_forest_model.pkl')
    else:
        rndforest.fit(x_train_aug, y_train_aug)

    # Prediction
    y_pred_rf = rndforest.predict(x_test)

    # Evaluation

    confusion_matrix_rf = confusion_matrix(final_y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf, display_labels=rndforest.classes_)
    disp_rf.plot()
    plt.title('RF Confusion Matrix - with "Defect Class"')
    plt.savefig(r'plots\confusion_matrix\without_delta_values\rf.png')
    plt.show()

    # Display the confusion matrix without class '0' -> no defect
    labels_rf = np.unique(final_y_test)[1:]
    confusion_matrix_xgb = confusion_matrix(final_y_test, y_pred_rf, labels=labels_rf)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_xgb, display_labels=labels_rf)
    disp_xgb.plot()
    plt.title('RF Confusion Matrix - without "Defect Class"')
    plt.savefig(r'plots\confusion_matrix\without_delta_values\rf_defects.png')
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
        dump(rndforest, r'models\without_delta_values\random_forest_model.pkl')

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

    if load_models == 1:
        xgb_model.load_model(r'models\without_delta_values\xgb_model.json')
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
    plt.savefig(r'plots\confusion_matrix\without_delta_values\xgboost.png')
    plt.show()

    # Display the confusion matrix without class '0' -> no defect
    labels_xgb = np.unique(final_y_test)[1:]
    confusion_matrix_xgb = confusion_matrix(final_y_test, y_pred_xgb, labels=labels_xgb)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_xgb, display_labels=labels_xgb)
    disp_xgb.plot()
    plt.title('XGBoost Confusion Matrix - without "Defect Class"')
    plt.savefig(r'plots\confusion_matrix\without_delta_values\xgboost_defects.png')
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
        xgb_model.save_model('models/without_delta_values/xgb_model.json')

    #######
    # SVM #
    #######

    print(f'\nStarting SVM...')

    svm_model = SVC(random_state=42, probability=True)

    if grid_search == 1:

        param_grid_svm = {
            'kernel': ['linear', 'poly'],
            'gamma': ['scale', 'auto']
        }

        grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring=scorer, verbose=4)
        grid_search_svm.fit(x_train_aug, y_train_aug)
        best_params_svm = grid_search_svm.best_params_
        best_recall_svm = grid_search_svm.best_score_
        print("Best parameters for SVM:", best_params_svm)
        print("Best recall for SVM:", best_recall_svm)

    if load_models == 1:
        svm_model = load(r'models\without_delta_values\svm_model.pkl')
    else:
        svm_model.fit(x_train_aug.values, y_train_aug)


    # Predict
    y_pred_svm = svm_model.predict(x_test)

    # Evaluation

    confusion_matrix_svm = confusion_matrix(final_y_test, y_pred_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_svm, display_labels=svm_model.classes_)
    disp_svm.plot()
    plt.title('SVM Confusion Matrix')
    plt.savefig(r'plots\confusion_matrix\without_delta_values\svm.png')
    plt.show()

    # Display the confusion matrix without class '0' -> no defect
    labels_svm = np.unique(final_y_test)[1:]
    confusion_matrix_svm = confusion_matrix(final_y_test, y_pred_rf, labels=labels_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_svm, display_labels=labels_svm)
    disp_svm.plot()
    plt.title('SVM Confusion Matrix - without "Defect Class"')
    plt.savefig(r'plots\confusion_matrix\without_delta_values\svm_defects.png')
    plt.show()

    recall_score_svm = recall_score(final_y_test, y_pred_svm, average='weighted', zero_division=1)
    print(f'Recall: {recall_score_svm:.6f}')

    precision_score_svm = precision_score(final_y_test, y_pred_svm, average='weighted', zero_division=1)
    print(f'Precision: {precision_score_svm:.6f}')

    logging.info("\nSVM Metrics:")
    logging.info(f"Recall: {recall_score_svm:.6f}")
    logging.info(f"Precision: {precision_score_svm:.6f}")

    # Save
    if load_models == 0:
        print("Saving SVM...")
        dump(svm_model, r'models\without_delta_values\svm_model.pkl')
        print("Saved!")

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

    if load_models == 1:
        catboost_model = CatBoostClassifier(loss_function='MultiClass', verbose=False)
        catboost_model.load_model(r'models\without_delta_values\catboost_model.cbm')
    else:
        catboost_model.fit(x_train_aug, y_train_aug)

    # Predict
    y_pred_cat = catboost_model.predict(x_test)

    # Evaluation

    confusion_matrix_cat = confusion_matrix(final_y_test, y_pred_cat)
    disp_cat = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cat, display_labels=catboost_model.classes_)
    disp_cat.plot()
    plt.title('CATBOOST Confusion Matrix')
    plt.savefig(r'plots\confusion_matrix\without_delta_values\catboost.png')
    plt.show()

    # Display the confusion matrix without class '0' -> no defect
    labels_cat = np.unique(final_y_test)[1:]
    confusion_matrix_cat = confusion_matrix(final_y_test, y_pred_rf, labels=labels_cat)
    disp_cat = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cat, display_labels=labels_cat)
    disp_cat.plot()
    plt.title('CATBOOST Confusion Matrix - without "Defect Class"')
    plt.savefig(r'plots\confusion_matrix\without_delta_values\catboost_defects.png')
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
        catboost_model.save_model(r'models\without_delta_values\catboost_model.cbm')
