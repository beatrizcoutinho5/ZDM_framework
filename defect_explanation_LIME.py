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
import lime
import lime.lime_tabular
from IPython.display import display
from IPython.display import HTML

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

rf_model_path = r'models\without_delta_values\binary\binary_random_forest_model.pkl'
xgb_model_path = r'models\without_delta_values\binary\binary_xgb_model.json'
catboost_model_path = r'models\without_delta_values\binary\binary_catboost_model.cbm'

# Load train and test data
x_train = pd.read_excel(
    r'data\split_train_test_data\without_delta_values\binary_data\binary_x_train_aug.xlsx')
y_train = pd.read_excel(
    r'data\split_train_test_data\without_delta_values\binary_data\binary_y_train_aug.xlsx')
x_test = pd.read_excel(
    r'data\split_train_test_data\without_delta_values\binary_data\binary_x_test.xlsx')
y_test = pd.read_excel(
    r'data\split_train_test_data\without_delta_values\binary_data\binary_y_test.xlsx')

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

# Explainer using Random Forest

explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train.values,
    feature_names=x_train.columns,
    class_names=['0', '1'],
    mode='classification',
    discretize_continuous=True
)

exp = explainer.explain_instance(
    x_test.iloc[26],
    rf_model.predict_proba
)

fig = exp.as_pyplot_figure(label=1)
fig.set_size_inches(20, 10)
plt.savefig(r'plots\lime\without_delta_values\binary\lime_rf.png')
plt.show()

# Explainer using CatBoost

explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train.values,
    feature_names=x_train.columns,
    class_names=['0', '1'],
    mode='classification',
    discretize_continuous=True
)

exp = explainer.explain_instance(
    x_test.iloc[26],
    catboost_model.predict_proba
)

fig = exp.as_pyplot_figure(label=1)
fig.set_size_inches(20, 10)
plt.savefig(r'plots\lime\without_delta_values\binary\lime_catboost.png')
plt.show()