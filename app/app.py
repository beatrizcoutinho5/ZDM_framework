from flask import Flask, request, render_template, url_for, request, jsonify, redirect
import numpy as np
import pandas as pd
import warnings

import numpy as np
import pandas as pd
import joblib
import warnings
import time
import random

from scipy.optimize import dual_annealing, minimize, basinhopping
from scipy.spatial.distance import minkowski
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from xgboost import XGBClassifier
from joblib import dump, load
from catboost import CatBoostClassifier
from flask_cors import CORS

warnings.filterwarnings("ignore", category=UserWarning)
model = load(r'models\binary\binary_random_forest_model.pkl')

data_path = r'binary_cleaned_data2022_2023_2024.xlsx'

df = pd.read_excel(data_path)
df = df.drop(["Recording Date", "Defect Code", "Group"], axis=1)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')


################################
######### PREDICTION #########
################################

def process_sample():
    # receive the sample data from the HTTP request
    sample_data = request.json

    # check if sample data is provided
    if not sample_data:
        return jsonify({'error': 'No sample data provided'}), 400

    # convert the sample to a NumPy array and reshape
    try:
        x = np.array(sample_data)
        x = x.reshape(1, -1)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    #
    reshaped_data = x.tolist()

    return reshaped_data


def sample_pre_processing(sample):
    sample = sample[0][0]

    if "Line Working?" in sample and sample["Line Working?"] == -1:
        return "Line is not working, sample not valid."

    # check for missing values in the sample
    has_missing_values = any(value is None for value in sample.values())
    if has_missing_values:
        return "The sample contains features with missing values."

    # Remove features that are not relevant for the model
    features_to_remove = ["Defect Group", "Defect Group Description", "Defect Description", "Pallet Code",
                          "Pallet Code Production Date", "Line Working?", "Humidity", "Temperature",
                          "Calculated Thermal Cycle Time", "Single Station Thermal Cycle Time",
                          "Double Station Thermal Cycle Time", "Jaw Clamping Time", "Suction Cup Pickup Time",
                          "Scratch Test", "Recording Date"]

    for feature in features_to_remove:
        if feature in sample:
            del sample[feature]

    # Transform categorical features into numerical representation
    columns_cat = ["GFFTT", "Finishing Top", "Finishing Down", "Reference Top", "Reference Down"]

    category_numerical_df = pd.read_excel('categorical_to_numeric_representations.xlsx')
    category_to_numerical_loaded = dict(
        zip(category_numerical_df['Category'], category_numerical_df['Numerical Value']))

    for feature in columns_cat:

        # print("Feature:", feature)
        # print("Original sample[feature] value:", sample[feature])

        # Check if the feature value exists in the mapping dictionary
        if sample[feature] in category_to_numerical_loaded:
            numerical_value = category_to_numerical_loaded[sample[feature]]
            sample[feature + '_cat'] = numerical_value
            # print("Numerical value:", numerical_value)

        del sample[feature]

    # Remove highly correlated features as they are not relevant for the model
    columns_remove_correlated = ["Thickness.1", "Lower Valve Temperature", "Upper Valve Temperature",
                                 "Liston 1 Speed.1", "Liston 2 Speed.1", "Floor 1.1", "Floor 2.1",
                                 "Bridge Platform.1", "Floor 1 Blow Time.1", "Floor 2 Blow Time.1",
                                 "Centering Table.1", "Finishing Down_cat", "Reference Down_cat"]

    for feature in columns_remove_correlated:
        if feature in sample:
            del sample[feature]

    # Width is assigned as an 'Object' type, so it needs to be typecasted into a numeric representation
    sample['Width'] = int(float(sample['Width']))

    return sample


def processed_sample(sample):
    reshaped_data = process_sample(sample)
    processed_sample = sample_pre_processing(reshaped_data)

    return processed_sample


def sample_defect_prediction_model(sample):
    # Extract values from the dictionary and convert them into a NumPy array
    sample_values = np.array(list(sample.values())).reshape(1, -1)

    prediction = model.predict(sample_values)

    return prediction


@app.route('/predict_defect', methods=['POST'])
def predict_defect(sample):
    processed_sample = processed_sample(sample)
    prediction = sample_defect_prediction_model(processed_sample)

    print(prediction)

    return render_template('result.html')


################################
######### OPTIMIZATION #########
################################

def predict_defect_score(sample):
    sample_values = sample.reshape(1, -1)
    prediction = model.predict_proba(sample_values)
    defect_score = prediction[:, 1]

    return defect_score

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


def feature_space_pre_processing(sample):

    column_names = [
        'Production Line', 'Production Order Code', 'Production Order Opening', 'Quantity', 'Length', 'Width',
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

    sample_values = []

    for i in range(len(sample)):
        sample_values.append(sample[i])

    sample_values = sample.flatten().tolist()

    print(sample_values)

    features_space = list(zip(column_names, sample_values))

    # intervals for the features that can be adjusted in real-time
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


    # Indices of the real time features in the features_space
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
    start_time = time.time()

    reshaped_data = process_sample()

    sample = sample_pre_processing(reshaped_data)

    sample = np.array(list(sample.values())).reshape(1, -1)

    initial_defect_score = predict_defect_score(sample)

    if initial_defect_score <= 0.1:

        print("Defect probability is too low, no need for optimization!")

        return "OKKKK"

    features_space, indices = feature_space_pre_processing(sample)

    target_defect_scores = [0.5]

    x0 = [
        410, 79430159686, 1, 30, 2800, 2070, 19, 120, 32, 10.2, 22, 0, 0, 23.6, 5.7,
        187, 188, 350, 0, 1400, 1200, 1, 0, 1, 2000, 4000, 40, 0, 0, 0, 0, 0, 1, 0,
        2200, 100, 0, 20, 50, 16, 1000, 0, 0, 0, 0, 1, 7, 7
    ]

    # Initialize variables to store the best result
    best_reduction_percentage = -float('inf')
    best_target_defect_score = None
    best_params = None
    best_mse = None
    best_params_selected = None
    best_elapsed_time = None
    best_final_defect_score = None

    # Iterate over each target defect score
    for target_defect_score in target_defect_scores:
        # Optimize parameters for the current target defect score
        current_params, current_mse = optimize_params(features_space, x0, target_defect_score)

        # Calculate final defect score and reduction percentage
        current_final_defect_score = predict_defect_score(current_params)
        current_reduction_percentage = (initial_defect_score - current_final_defect_score) * 100

        # Update best result if the current reduction percentage is higher
        if current_reduction_percentage > best_reduction_percentage:
            best_reduction_percentage = current_reduction_percentage
            best_target_defect_score = target_defect_score
            best_params = current_params
            best_mse = current_mse
            best_final_defect_score = current_final_defect_score
            best_params_selected = current_params[indices]
            best_elapsed_time = time.time() - start_time

    # Print the best optimization results
    if best_reduction_percentage <= 0:
        print("Parameters can't be optimized :(")
    else:
        print('\n---- Optimization Results ----')
        print('\nTarget Defect Score:   ', best_target_defect_score)
        print('Best Parameters:    ', best_params_selected)
        print('Initial Defect Score:  ', initial_defect_score)
        print('Final Defect Score:    ', best_final_defect_score)
        print(f'Reduced Defect Score in {best_reduction_percentage}%')
        print('Elapsed Time (in seconds):    ', round(best_elapsed_time, 2))
        print('MSE:                ', best_mse)

    return "OK :)"


if __name__ == '__main__':
    app.run(debug=True)
