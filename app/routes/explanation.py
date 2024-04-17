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

from .dataprocessing import process_sample, sample_pre_processing
from __main__ import app

# model = CatBoostClassifier()
# model.load_model(r'models\binary\binary_catboost_model.cbm')

model = load(r'models\binary\binary_random_forest_model.pkl')

def init_explanation_routes(app):
    @app.route('/shap_explainer', methods=['POST'])
    def shap_explainer_route():
        return shap_explainer()

def shap_explainer():

    reshaped_data = process_sample()
    sample = sample_pre_processing(reshaped_data)

    sample_values = np.array(list(sample.values())).reshape(1, -1)  # Reshape to have one row
    print(f'Sample Values: {sample_values}')

    sample_features = list(map(str.strip, sample.keys()))
    print(f'Sample Features: {sample_features}')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_values)

    fig, ax = plt.subplots(figsize=(20, 10))
    shap.summary_plot(shap_values, features=sample_values, feature_names=sample_features,
                      plot_type='bar', show=False)

    plt.show()

    return "OK!"


