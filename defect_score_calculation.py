import os
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

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

warnings.filterwarnings("ignore")

data_path = r'data\clean_data\binary_cleaned_data_with_deltavalues_2022_2023_2024.xlsx'
data = pd.read_excel(data_path)

rf_model_path = r'models\with_delta_values\binary\NOTNORM_binary_random_forest_model.pkl'
xgb_model_path = r'models\with_delta_values\binary\NOTNORM_binary_xgb_model.json'
catboost_model_path = r'models\with_delta_values\binary\NOTNORM_binary_catboost_model.cbm'

# Select row 252 for test
data = data.loc[[252]]

# Duplicate Row
new_row = data.copy()
new_row['Thermal Cycle Time'] = 26
new_row['Pressure'] = 280
new_row['Lower Plate Temperature'] = 178
new_row['Upper Plate Temperature'] = 182


data = data.append(new_row, ignore_index=True)
data = data.drop(["Recording Date", "Defect Code"], axis=1)

# Random Forest
rf_model = joblib.load(rf_model_path)
rf_prob = rf_model.predict_proba(data)
print("rf done")

# XGBoost
xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)
xgb_prob = xgb_model.predict_proba(data)
print("xgb done")

# CatBoost
catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)
catboost_prob = catboost_model.predict_proba(data)
print("catb done")

avg_prob = np.mean([rf_prob, xgb_prob, catboost_prob], axis=0)
avg_defect_score = avg_prob[:, 1]  # Using the probabilities of class 1 as defect score

df_results = pd.DataFrame({
    'Avg Defect Score': avg_defect_score
})

df_results = pd.concat([df_results, data], axis=1)
df_results.to_excel(r'verify_optim.xlsx')