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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from .dataprocessing import process_sample, sample_pre_processing
from __main__ import app

# model = load(r'models\binary\binary_random_forest_model.pkl')
model = CatBoostClassifier ()
model.load_model(r'models\binary\binary_catboost_model.cbm')

data_path = r'binary_cleaned_data2022_2023_2024.xlsx'
df = pd.read_excel(data_path)
df = df.drop(["Recording Date", "Defect Code", "Group"], axis=1)

# the names of the features that must be present after the sample is processed
column_names = [
    'Production Line', 'Production Order Code', 'Production Order Opening', 'Length', 'Width',
    'Thickness', 'Lot Size', 'Cycle Time', 'Mechanical Cycle Time', 'Thermal Cycle Time',
    'Control Panel with Micro Stop',
    'Control Panel Delay Time', 'Sandwich Preparation Time', 'Carriage Time', 'Lower Plate Temperature',
    'Upper Plate Temperature', 'Pressure', 'Roller Start Time', 'Liston 1 Speed', 'Liston 2 Speed', 'Floor 1',
    'Floor 2', 'Bridge Platform', 'Floor 1 Blow Time', 'Floor 2 Blow Time', 'Centering Table',
    'Conveyor Belt Speed Station 1', 'Quality Inspection Cycle', 'Conveyor Belt Speed Station 2',
    'Transverse Saw Cycle',
    'Right Jaw Discharge', 'Left Jaw Discharge', 'Simultaneous Jaw Discharge', 'Carriage Speed', 'Take-off Path',
    'Stacking Cycle', 'Lowering Time', 'Take-off Time', 'High Pressure Input Time', 'Press Input Table Speed',
    'Scraping Cycle', 'Paper RC', 'Paper VC', 'Paper Shelf Life', 'GFFTT_cat', 'Finishing Top_cat',
    'Reference Top_cat'
]

def init_optimization_routes(app):
    @app.route('/optimize_defect_score', methods=['POST'])
    def optimize_defect_score_route():
        return optimize_defect_score()


# defect score = model probability
def predict_defect_score(sample):

    sample_values = sample.reshape(1, -1)
    prediction = model.predict_proba(sample_values)
    defect_score = prediction[:, 1]

    return defect_score


# mse fitness function for the optimizer
def fitness_function(x, target_defect_score, features_space):
    x_concat = build_feature_array(x, features_space)
    current_defect_score = predict_defect_score(x_concat)

    return mean_squared_error(current_defect_score, [target_defect_score])


def build_feature_array(x, features_space):
    x_concat = np.zeros(len(features_space))
    x_list = list(x)
    for i, v in enumerate(features_space):
        if type(v[1]) != tuple:
            x_concat[i] = v[1]
        else:
            x_concat[i] = x_list.pop(0)
    return x_concat


# defect score optimization using dual annealing
def optimize_params(features_space, x0, target_defect_score):
    for i, v in enumerate(features_space):
        if v[1] is None:
            features_space[i][1] = (df[v[0]].min(), df[v[0]].max())

    nff_idx, bounds = zip(*[(i, v[1]) for i, v in enumerate(features_space) if type(v[1]) == tuple])
    x0_filtered = [v for i, v in enumerate(x0) if i in set(nff_idx)]

    result = dual_annealing(
        func=fitness_function,
        x0=x0_filtered,
        bounds=bounds,
        args=[target_defect_score, features_space],
        maxfun=1e3,
        seed=16
    )

    best_params = build_feature_array(result.x, features_space)
    mse = result.fun

    return best_params, mse


# features space pre-processing
def feature_space_pre_processing(sample):
    sample_values = []

    for i in range(len(sample)):
        sample_values.append(sample[i])

    sample_values = sample.flatten().tolist()


    # save the sample feature values along with the feature names
    features_space = list(zip(column_names, sample_values))

    # intervals for the features that can be adjusted in real-time
    # the rest of the features can't be adjusted
    intervals = {
        'Thermal Cycle Time': (10, 150),
        'Pressure': (250, 350),
        'Lower Plate Temperature': (160, 210),
        'Upper Plate Temperature': (160, 210)
    }

    # updates the values (bounds) for the real time features in the features_space
    for i, (f, v) in enumerate(features_space):
        if f in intervals:
            features_space[i] = (f, intervals[f])

    # indices of the real time features in the features_space (used to only print the relevant features later)
    thermal_cycle_time_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Thermal Cycle Time'][
        0]
    pressure_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Pressure'][0]
    lower_plate_temp_index = \
    [i for i, (feature, _) in enumerate(features_space) if feature == 'Lower Plate Temperature'][0]
    upper_plate_temp_index = \
    [i for i, (feature, _) in enumerate(features_space) if feature == 'Upper Plate Temperature'][0]

    indices = [thermal_cycle_time_index, pressure_index, lower_plate_temp_index, upper_plate_temp_index]

    return features_space, indices


@app.route('/optimize_defect_score', methods=['POST'])
def optimize_defect_score():
    # start timer to record the time the optimization took, from receiving the sample to providing
    # a parameteres adjustment suggestion

    start_time = time.time()

    # sample pre-processing (receiving a raw sample)
    reshaped_data = process_sample()
    sample = sample_pre_processing(reshaped_data)

    sample = {feature: sample.get(feature) for feature in column_names}
    sample = np.array(list(sample.values())).reshape(1, -1)

    initial_defect_score = predict_defect_score(sample)

    # if the defect score is under 10%, an optimization is not needed
    if initial_defect_score <= 0.1:
        print("Defect probability is too low, no need for optimization!")

        return "OK"

    # obtaining features_space
    features_space, indices = feature_space_pre_processing(sample)

    initial_parameters = [sample[0][index] for index in indices]

    # defining the target defect score for the optimizer
    target_defect_scores = [0.01, 0.5]

    # reference sample to start the optimization (very low defect score)
    x0 = [
        410, 79430159686, 1, 30, 2800, 2070, 19, 120, 32, 10.2, 22, 0, 0, 23.6, 5.7,
        187, 188, 350, 0, 1400, 1200, 1, 0, 1, 2000, 4000, 40, 0, 0, 0, 0, 0, 1, 0,
        2200, 100, 0, 20, 50, 16, 1000, 0, 0, 0, 0, 1, 7, 7
    ]

    # since we are experimenting with different target defect scores, after we calculate the optimization considering
    # each targe, we need to store the required variables to then return only the best result

    best_reduction_percentage = -float('inf')
    best_target_defect_score = None
    best_mse = None
    best_params_selected = None
    best_elapsed_time = None
    best_final_defect_score = None

    for target_defect_score in target_defect_scores:

        current_params, current_mse = optimize_params(features_space, x0, target_defect_score)

        current_final_defect_score = predict_defect_score(current_params)
        current_reduction_percentage = (initial_defect_score - current_final_defect_score) * 100

        # update best result if the current reduction percentage is higher

        if current_reduction_percentage > best_reduction_percentage:
            best_reduction_percentage = current_reduction_percentage
            best_target_defect_score = target_defect_score
            best_mse = current_mse
            best_final_defect_score = current_final_defect_score
            best_params_selected = current_params[indices]
            best_elapsed_time = time.time() - start_time

    # if the algorithm wasn't able to reduce the initial defect score return an error message
    if best_reduction_percentage <= 0:
        print("Parameters can't be optimized :(")

    else:
        # print the best optimization results

        print('---- Optimization Results ----')
        print('\nTarget Defect Score:   ', best_target_defect_score)
        print('Initial Parameters:    ', initial_parameters)
        print('Best Parameters:    ', best_params_selected.round(2))
        print('Initial Defect Score:  ', initial_defect_score)
        print('Final Defect Score:    ', best_final_defect_score)
        print('Reduced Defect Score in:    ', best_reduction_percentage, '%')
        print('Elapsed Time (in seconds):    ', round(best_elapsed_time, 2))
        print('MSE:                ', best_mse.round(3))

    return "OK :)"