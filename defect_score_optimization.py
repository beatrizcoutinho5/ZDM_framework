import os
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


# Probability predictions
def get_model_predictions(model, sample):
    return model.predict_proba(sample)[:, 1]

rf_model = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models'
xgb_model = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models'
svm_model = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models'
catboost_model = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\models'

x_test = pd.read_excel(r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\x_test.xlsx')

rf_probs = get_model_predictions(rf_model, x_test)
xgb_probs = get_model_predictions(xgb_model, x_test)
svm_probs = get_model_predictions(svm_model, x_test)
cat_probs = get_model_predictions(catboost_model, x_test)


# Averaging the probability predictions from all models

avg_probs = np.mean([xgb_probs, cat_probs, rf_probs, ], axis=0)

# Defect score calculation and saving

x_test_with_defect_scores = x_test.copy()
x_test_with_defect_scores['Defect_Score'] = avg_probs

output_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\data\x_test_with_defect_scores.xlsx'
x_test_with_defect_scores.to_excel(output_path, index=False)

print(avg_probs)