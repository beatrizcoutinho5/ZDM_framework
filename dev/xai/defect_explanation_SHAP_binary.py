import os
import shap
import joblib
import random
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Suppress InconsistentVersionWarning
warnings.filterwarnings("ignore", category=UserWarning)

rf_model_path = r'../prediction/models/binary/binary_random_forest_model.pkl'
xgb_model_path = r'../prediction/models/binary/binary_xgb_model.json'
catboost_model_path = r'../prediction/models/binary/binary_catboost_model.cbm'

# Load train and test data
x_train = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_x_train.xlsx')
y_train = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_y_train.xlsx')
x_test = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_x_test.xlsx')
y_test = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_y_test.xlsx')

# Reduce data size to 1% of original (for the test to be faster since SHAP takes a long time)
num_samples = int(0.0005 * len(x_train))
indices_optimize = range(num_samples)

x_train = x_train.iloc[indices_optimize]
y_train = y_train.iloc[indices_optimize]
x_test = x_test.iloc[indices_optimize]
y_test = y_test.iloc[indices_optimize]
print("Loaded data!")

# Load trained models
rf_model = joblib.load(rf_model_path)

xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)

catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)

print("Loaded models!")

feature_names = x_test.columns
feature_names = feature_names.astype(str)
class_names = y_test['Defect Code'].unique()
fig_size = (16, 9)

#################
# Random Forest #
#################

# print(f'------- Random Forest -------')
#
# print("Calculating and Computing SHAP values...")
# rf_explainer = shap.TreeExplainer(rf_model)
# rf_shap_values = rf_explainer.shap_values(x_test)
# print(f'rf_shap_values : {rf_shap_values.shape}')
#
# print("Plotting and saving SHAP summary plots...")
#
# shap_values_for_class = rf_shap_values[:, :, 1]
#
# mean_abs_shap_values = np.mean(np.abs(shap_values_for_class), axis=0)
#
# top_features_indices = np.argsort(mean_abs_shap_values)[-10:]
#
# top_features_shap_values = shap_values_for_class[:, top_features_indices]
# top_features = x_test.iloc[:, top_features_indices]
#
# fig, ax = plt.subplots(figsize=fig_size)
# shap.summary_plot(top_features_shap_values, features=top_features, feature_names=top_features.columns, plot_type='bar',
#                   show=False)
#
# ax.set_title(f"SHAP Summary Plot for Defect Class")
# plt.savefig(os.path.join(r'../plots/shap/binary/random_forest', f"shap_summary_plot_defect_rf.png"), dpi=300, bbox_inches='tight')
# plt.show()
# #
# # ################
# # #   XGBoost    #
# # ################
# #
# # print(f'------- XGBoost -------')
# #
# # print("Calculating and Computing SHAP values for XGBoost...")
# # dtest = DMatrix(x_test)
# # xgb_explainer = shap.TreeExplainer(xgb_model)
# # xgb_shap_values = xgb_explainer.shap_values(dtest)
# #
# # print("Plotting and saving SHAP summary plots for XGBoost...")
# #
# # shap_values_for_class = xgb_shap_values[1]
# #
# # shap_values_for_class_reshaped = shap_values_for_class.reshape(len(xgb_shap_values), -1)
# #
# # fig, ax = plt.subplots(figsize=fig_size)
# # shap.summary_plot(shap_values_for_class_reshaped, features=x_test, feature_names=x_test.columns, plot_type='bar',
# #                   show=False)
# #
# # ax.set_title(f"SHAP Summary Plot for Defect Class")
# # plt.savefig(os.path.join(r'..\plots\shap\binary\xgboost', f"shap_summary_plot_defect_xgb.png"), dpi=300, bbox_inches='tight')
# # plt.show()
# #
# #
################
#   CatBoost   #
################

print(f'------- CatBoost -------')

print("Calculating and Computing SHAP values for CatBoost...")
catboost_explainer = shap.Explainer(catboost_model)

testar = x_test.iloc[54]

catboost_shap_values = catboost_explainer.shap_values(testar)

print("Plotting and saving SHAP summary plots for CatBoost...")

shap_values_for_class = catboost_shap_values[:, 1]
print(shap_values_for_class)

shap_values_for_class_reshaped = shap_values_for_class.reshape(len(catboost_shap_values), -1)

fig, ax = plt.subplots(figsize=fig_size)

# shap.summary_plot(shap_values_for_class_reshaped, features=x_test, feature_names=x_test.columns, plot_type='bar',
#                   show=False)
#
# ax.set_title(f"SHAP Summary Plot for Defect Class")
# plt.savefig(os.path.join(r'..\plots\shap\binary\catboost', f"shap_summary_plot_defect_catboost.png"), dpi=300, bbox_inches='tight')

shap.plots.waterfall(shap_values_for_class_reshaped, max_display=10, show=True)

plt.show()