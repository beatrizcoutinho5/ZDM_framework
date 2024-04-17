from catboost import CatBoostClassifier
from flask import Flask, request, render_template, url_for, request, jsonify, redirect
import numpy as np
import pandas as pd
import joblib
import warnings
import time

from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from catboost import CatBoostClassifier

from .dataprocessing import process_sample, sample_pre_processing
from __main__ import app

# model = load(r'models\binary\binary_random_forest_model.pkl')

model = CatBoostClassifier()
model.load_model(r'models\binary\binary_catboost_model.cbm')


def init_prediction_routes(app):
    @app.route('/predict_defect', methods=['POST'])
    def predict_defect_route():
        sample = process_sample()
        processed_sample = sample_pre_processing(sample)
        prediction = sample_defect_prediction_model(processed_sample)

        print(f'Prediction: {prediction}')

        return "OKKKKKKKKK <3"


def sample_defect_prediction_model(sample):
    # convert into a NumPy array
    sample_values = np.array(list(sample.values())).reshape(1, -1)

    # defect prediction model
    prediction = model.predict(sample_values)

    return prediction


