import time
import numpy as np
import pandas as pd

from joblib import load
from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model(r'models\binary\binary_catboost_model.cbm')
# model = load(r'models\binary\binary_random_forest_model.pkl')

# The names of the features that must be present after the sample is processed
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


# Defect score = Model probability
def predict_defect_score(sample):

    sample_values = sample.reshape(1, -1)
    prediction = model.predict_proba(sample_values)
    defect_score = prediction[:, 1]

    return defect_score


# MSE fitness function for the optimizer
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


# Defect score optimization using Dual Annealing
def optimize_params(features_space, x0, target_defect_score):

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


# Features space pre-processing
def feature_space_pre_processing(sample):

    sample_values = []

    for i in range(len(sample)):
        sample_values.append(sample[i])

    sample_values = sample.flatten().tolist()

    # Save the sample feature values along with the feature names
    features_space = list(zip(column_names, sample_values))

    # Intervals for the features that can be adjusted in real-time
    # the rest of the features can't be adjusted
    intervals = {
        'Thermal Cycle Time': (0, 2300),
        'Pressure': (0, 2300),
        'Lower Plate Temperature': (0, 2300),
        'Upper Plate Temperature': (0, 2300),
        'Cycle Time': (0, 300),
        'Mechanical Cycle Time': (0, 300),
        'Carriage Speed': (0, 2300),
        'Press Input Table Speed': (0, 2300),
        'Scraping Cycle': (0, 23),
        'Transverse Saw Cycle': (0, 2300),
    }

    real_time_features = ['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature', 'Upper Plate Temperature',
                          'Cycle Time', 'Mechanical Cycle Time', 'Carriage Speed', 'Press Input Table Speed',
                          'Scraping Cycle', 'Transverse Saw Cycle']

    # Indices of the real time features in the features_space
    indices_dict = {}
    for feature in real_time_features:
        indices_dict[feature] = [i for i, (feat, _) in enumerate(features_space) if feat == feature][0]

    indices = list(indices_dict.values())

    # Update the optimization bounds for the real time features
    for feature, value in features_space:

        if feature in real_time_features:

            # Get the current value of the feature
            min_val, max_val = intervals[feature]

            # The optimization range is the actual value +/- 20%, to allow a realistic adjustment
            adjustment = value * 0.2

            new_min = max(min_val, value - adjustment)
            new_max = min(max_val, value + adjustment)

            # If the current/actual value is zero, the maximum is set to 10% of the upper bound defined in 'intervals'
            if value == 0 or new_max == 0:
                new_max = 0.1 * max_val

            # Ensure the adjusted values stay within the interval bounds
            new_min = max(new_min, min_val)
            new_max = min(new_max, max_val)

            # If the current/actual value is larger than the defined maximum value, the
            # upper bound is set to the current value
            if value > max_val:
                new_max = value
                new_min = max(min_val, new_max - adjustment)

            index = features_space.index((feature, value))
            features_space[index] = (feature, (new_min, new_max))

    return features_space, indices


# @app.route('/optimize_defect_score')
def optimize_defect_score(sample):

    # Start timer to record the time the optimization took, from receiving the sample to providing
    # a parameteres adjustment suggestion
    start_time = time.time()

    sample = {feature: sample.get(feature) for feature in column_names}
    sample = np.array(list(sample.values())).reshape(1, -1)

    initial_defect_score = predict_defect_score(sample)

    # Obtaining features_space
    features_space, indices = feature_space_pre_processing(sample)

    # Get sample's initial real-time features values
    initial_parameters = [sample[0][index] for index in indices]
    current_tct = round(initial_parameters[0])
    current_pressure = round(initial_parameters[1])
    current_lpt = round(initial_parameters[2])
    current_upt = round(initial_parameters[3])
    current_ct = round(initial_parameters[4])
    current_mct = round(initial_parameters[5])
    current_cs = round(initial_parameters[6])
    current_pits = round(initial_parameters[7])
    current_sc = round(initial_parameters[8])
    current_tsc = round(initial_parameters[9])

    # If the defect score is under 10%, an optimization is not needed
    if initial_defect_score <= 0.1:
        return "Defect probability is too low, no need for optimization!", "-", "-", current_tct, current_pressure, current_lpt, current_upt, current_ct, current_mct, current_cs, current_pits, current_sc, current_tsc

    # Defining the target defect scores for the optimizer
    # target_defect_scores = [0.01, 0.5]
    target_defect_scores = [0]

    # Reference sample to start the optimization (very low defect score)
    x0 = [
        410, 79430159686, 1, 30, 2800, 2070, 19, 120, 32, 10.2, 22, 0, 0, 23.6, 5.7,
        187, 188, 350, 0, 1400, 1200, 1, 0, 1, 2000, 4000, 40, 0, 0, 0, 0, 0, 1, 0,
        2200, 100, 0, 20, 50, 16, 1000, 0, 0, 0, 0, 1, 7, 7
    ]

    # Since we are experimenting with different target defect scores, after calculating the optimization considering
    # each target, we need to store the required variables to then return only the best result

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

        # Update best result if the current reduction percentage is higher
        if current_reduction_percentage > best_reduction_percentage:
            best_reduction_percentage = current_reduction_percentage
            best_target_defect_score = target_defect_score
            best_mse = current_mse
            best_final_defect_score = current_final_defect_score
            best_params_selected = current_params[indices]
            best_elapsed_time = time.time() - start_time

    # If the algorithm wasn't able to reduce the initial defect score
    if best_reduction_percentage <= 0:

        initial_defect_score_p = initial_defect_score[0] * 100

        return "Parameters can't be optimized!", initial_defect_score_p, "0", current_tct, current_pressure, current_lpt, current_upt, current_ct, current_mct, current_cs, current_pits, current_sc, current_tsc

    else:

        # Print the best optimization results
        print('\n**** Optimization Results ****')
        print('Target Defect Score:   ', best_target_defect_score)
        print('Initial Parameters:    ', initial_parameters)
        print('Best Parameters:    ', best_params_selected.round(0))
        print('Initial Defect Score:  ', initial_defect_score)
        print('Final Defect Score:    ', best_final_defect_score)
        print('Reduced Defect Score in:    ', best_reduction_percentage, '%')
        print('Elapsed Time (in seconds):    ', round(best_elapsed_time, 2))
        print('MSE:                ', best_mse.round(3))

    # Return final defect score
    best_final_defect_score = best_final_defect_score * 100
    best_final_defect_score = np.round(best_final_defect_score, 1)
    best_final_defect_score = best_final_defect_score[0]

    # Return the difference between the defect score before and after optimization
    best_reduction_percentage = np.round(best_reduction_percentage, 1)
    best_reduction_percentage = best_reduction_percentage[0]

    # Return the adjusted values of the real-time features that allow the defect score reduce
    tct_after_optim = round(best_params_selected[0], 1)  # Thermal Cycle Time
    pressure_after_optim = round(best_params_selected[1], 1)  # Pressure
    lpt_after_optim = round(best_params_selected[2], 1)  # Lower Plate Temperature
    upt_after_optim = round(best_params_selected[3], 1)  # Upper Plate Temperature
    ct_after_optim = round(best_params_selected[4], 1)  # Cycle Time
    mct_after_optim = round(best_params_selected[5], 1)  # Mechanical Cycle Time
    cs_after_optim = round(best_params_selected[6], 1)  # Carriage Speed
    pits_after_optim = round(best_params_selected[7], 1)  # Press Input Table Speed
    sc_after_optim = round(best_params_selected[8], 1)  # Scraping Cycle
    tsc_after_optim = round(best_params_selected[9], 1)  # Transverse Saw Cycle

    return "Defect Probability After Optimization", best_final_defect_score, best_reduction_percentage, tct_after_optim, pressure_after_optim, lpt_after_optim, upt_after_optim, ct_after_optim, mct_after_optim, cs_after_optim, pits_after_optim, sc_after_optim, tsc_after_optim
