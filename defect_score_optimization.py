import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from scipy.optimize import dual_annealing
from sklearn.svm import SVC
from datetime import datetime
from sklearn import preprocessing
from xgboost import XGBClassifier, DMatrix
from joblib import dump, load
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

rf_model_path = r'models\with_delta_values\binary\NOTNORM_binary_random_forest_model.pkl'
xgb_model_path = r'models\with_delta_values\binary\NOTNORM_binary_xgb_model.json'
svm_model_path = r'models\with_delta_values\binary\NOTNORM_binary_svm_model.pkl'
catboost_model_path = r'models\with_delta_values\binary\NOTNORM_binary_catboost_model.cbm'

# Load train and test data

x_train = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_x_test.xlsx')
y_train = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_y_test.xlsx')

x_test = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_x_test.xlsx')
y_test = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_y_test.xlsx')

# Load trained models
print("loaded data")
# Random Forest

rf_model = joblib.load(rf_model_path)
rf_prob = rf_model.predict_proba(x_test)
print("rf done")
# XGBoost

xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)
xgb_prob = xgb_model.predict_proba(x_test)
print("xgb done")
# SVM

svm_model = joblib.load(svm_model_path)
svm_prob = svm_model.predict_proba(x_test)
print("svm done")
# CatBoost

catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)
catboost_prob = catboost_model.predict_proba(x_test)
print("catb done")

# Defect Score will be the average of the probability returned by each model

avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

df_results = pd.DataFrame({
    'Defect Code': y_test['Defect Code'],
    'Avg Defect Score': avg_defect_score
})

df_results = pd.concat([df_results, x_test.reset_index(drop=True)], axis=1)

# Features that can be "changed" in real time
real_time_features = ['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature',
                      'Upper Plate Temperature']

# Search Space
# Defining the parameter boudaries for each features
features_space = [
    ['Thermal Cycle Time', (10, 120)],
    ['Pressure', (280, 350)],
    ['Lower Plate Temperature', (165, 202)],
    ['Upper Plate Temperature', (169, 195)]
]

for feature_name in x_test.columns:
    if feature_name not in ['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature', 'Upper Plate Temperature']:
        features_space.append([feature_name, None])

print(features_space)


def fitness_function(y_target, y_predicted):
    return (y_target - y_predicted) ** 2
