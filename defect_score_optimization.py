import os
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error
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
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_x_train_aug.xlsx')
y_train = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_y_train_aug.xlsx')
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

avg_prob = np.mean([rf_prob, xgb_prob, svm_prob, catboost_prob], axis=0)
avg_defect_score = avg_prob[:, 1]  # Using the probabilities of class 1 as defect score

df_results = pd.DataFrame({
    'Defect Code': y_test['Defect Code'],
    'Avg Defect Score': avg_defect_score
})

df_results.to_excel(r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_x_test_with_defect_scores.xlsx')
#
# df_results = pd.concat([df_results, x_test], axis=1)
#
# # Filtering the data to find a good "reference" sample to test the optimization
# filtered_results = df_results.query('`Defect Code` == 0 and `Avg Defect Score` < 0.15')
# x0 = filtered_results.iloc[0]
# print("X0 selected.")
#
# target_defect_score = 0.1
#
# # Features that can be "changed" in real time
# real_time_features = ['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature',
#                       'Upper Plate Temperature']
#
# # Search Space
# # Defining the parameter boudaries for real time features
# real_time_features_bounds = [
#     ['Thermal Cycle Time', (10, 120)],
#     ['Pressure', (280, 350)],
#     ['Lower Plate Temperature', (165, 202)],
#     ['Upper Plate Temperature', (169, 195)]
# ]
#
# # Data to be optimized (test)
# # to be faster i'm only using 10% of the original test data
# num_samples = int(0.1 * len(x_test))
# indices_optimize = np.random.choice(len(x_test), num_samples, replace=False)
# x_optimize = x_test.iloc[indices_optimize]
#
#
# # function to obtain the defect score
# def defect_score(x):
#
#     print("Shape of x:", x.shape)  # Print the shape of x
#
#
#
#     rf_prob = rf_model.predict_proba(x)
#     xgb_prob = xgb_model.predict_proba(x)
#     svm_prob = svm_model.predict_proba(x)
#     catboost_prob = catboost_model.predict_proba(x)
#
#     avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], svm_prob[:, 1], catboost_prob[:, 1]], axis=0)
#
#     return avg_defect_score
# def build_feature_array(x, real_time_features_bounds):
#     x_concat = np.zeros(len(real_time_features_bounds))
#     x_list = list(x)
#     for i, v in enumerate(real_time_features_bounds):
#         # appends the fixed feature values
#         if type(v[1]) != tuple:
#             x_concat[i] = v[1]
#         # appends the non fixed feature values
#         else:
#             x_concat[i] = x_list.pop(0)
#     # returns the results
#     return x_concat
#
# def fitness_function(x, target_defect_score, real_time_features_bounds):
#
#     x_concat = build_feature_array(x, real_time_features_bounds)
#
#     current_defect_score = defect_score(x_concat)
#     print(f'current_defect_score: {current_defect_score}')
#     return mean_squared_error(current_defect_score, target_defect_score)
#
# # Define callback function for dual_annealing
# def dual_annealing_callback(x, f, context):
#     print('non-fixed params: {0}, defect score: {1}'.format(x.round(2), f.round(3)))
#
# # Modify optimize_params function to accept callback function
# def optimize_params(real_time_features_bounds, x0, target_defect_score, cb=dual_annealing_callback):
#
#     bounds = [bounds_tuple[1] for bounds_tuple in real_time_features_bounds]
#
#     result = dual_annealing(
#         func=fitness_function,
#         bounds=bounds,
#         args=(target_defect_score, real_time_features_bounds),
#         callback=cb  # Pass the callback function here
#     )
#
#     best_params = result.x
#     mse = result.fun
#     # returns the results
#     return best_params, mse
#
#
# # Adjust the call to optimize_params to pass necessary arguments
# print("start optimization...")
# best_params, mse = optimize_params(real_time_features_bounds, x0, target_defect_score)
# final_defect_score = defect_score(best_params)
#
# # prints the results
# print('\n** Optimization Results **')
# print('Target Score:   ', target_defect_score)
# print('Previous Parameters:', x0)
# print('Best Parameters:    ', best_params.round(2))
# print('Final Whiteness:    ', final_defect_score.round(2))
# print('MSE:                ', mse.round(3))