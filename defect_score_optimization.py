import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


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

rf_model_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models\with_delta_values\binary_random_forest_model.pkl'
xgb_model_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models\with_delta_values\binary_xgb_model.json'
svm_model_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models\with_delta_values\binary_svm_model.pkl'
catboost_model_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models\with_delta_values\binary_catboost_model.cbm'

x_test = pd.read_excel(r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_x_test.xlsx')
y_test = pd.read_excel(r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_y_test.xlsx')

# Random Forest

rf_model = joblib.load(rf_model_path)
rf_prob = rf_model.predict_proba(x_test.values)

# XGBoost

xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)
xgb_prob = xgb_model.predict_proba(x_test)

# SVM

svm_model = joblib.load(svm_model_path)
svm_prob = svm_model.predict_proba(x_test.values)

# CatBoost

catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)
catboost_prob = catboost_model.predict_proba(x_test)


# Defect Score

avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

model_probs = pd.DataFrame(columns=['Defect Code', 'RF Score', 'XGB Score', 'SVM Score', 'CATBOOST Score', 'Avg. Defect Score'])
# model_probs = pd.DataFrame(columns=['Defect Code', 'RF Score', 'XGB Score', 'CATBOOST Score', 'Avg. Defect Score'])

model_probs['Defect Code'] = y_test['Defect Code']
model_probs['RF Score'] = rf_prob[:, 1]
model_probs['XGB Score'] = xgb_prob[:, 1]
model_probs['SVM Score'] = svm_prob[:, 1]
model_probs['CATBOOST Score'] = catboost_prob[:, 1]
model_probs['Avg. Defect Score'] = avg_defect_score

model_probs.to_excel(r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\split_train_test_data\with_delta_values\binary_x_test_with_defect_scores.xlsx')
print("Saved Defect Scores to Excel!")





