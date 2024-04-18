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
from .prediction import sample_defect_prediction_model
from __main__ import app

# model = CatBoostClassifier()
# model.load_model(r'models\binary\binary_catboost_model.cbm')

model = load(r'models\binary\binary_random_forest_model.pkl')

x_train = pd.read_excel(
    r'routes\binary_x_train_aug.xlsx')

def init_explanation_routes(app):
    @app.route('/shap_explainer', methods=['POST'])
    def shap_explainer_route():
        return shap_explainer()

def shap_explainer():

    data = process_sample()
    sample = sample_pre_processing(data)

    # prediction
    prediction = sample_defect_prediction_model(sample)

    rf_explainer = shap.TreeExplainer(model)

    sample_keys = list(sample.keys())

    sample = list(sample.values())
    sample = np.array(sample)

    rf_shap_values = rf_explainer.shap_values(sample)

    shap_values_for_class = rf_shap_values[:, 1]

    shap_values_for_class = shap_values_for_class.reshape(-1, 1)
    shap_values_for_class = shap_values_for_class.transpose()

    sample = sample.reshape(-1, len(sample_keys))

    fig, ax = plt.subplots(figsize=(20, 20))
    shap.summary_plot(shap_values_for_class, features=sample, feature_names=sample_keys, plot_type='bar',
                      show=False)
    plt.show()




    # ### TESTE com o LIME
    #
    # # Explainer using Random Forest
    #
    # explainer = lime.lime_tabular.LimeTabularExplainer(
    #     x_train.values,
    #     feature_names=x_train.columns,
    #     class_names=['0', '1'],
    #     mode='classification',
    #     discretize_continuous=True
    # )
    #
    # sample_values = [sample[key] for key in x_train.columns]
    #
    # exp = explainer.explain_instance(
    #     sample_values,
    #     model.predict_proba
    # )
    #
    # fig = exp.as_pyplot_figure(label=1)
    # fig.set_size_inches(20, 10)
    # plt.show()

    return "OK!"


