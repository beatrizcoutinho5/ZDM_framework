import numpy as np
import pandas as pd
import joblib
import warnings

from scipy.optimize import dual_annealing, minimize, basinhopping
from scipy.spatial.distance import minkowski
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

dual_annealing_optim = 1
powell_optim = 0
nelder_mead_optim = 0
basinhopping_optim = 0

warnings.filterwarnings("ignore")

data_path = r'data\clean_data\binary_cleaned_data_with_deltavalues_2022_2023_2024.xlsx'

df = pd.read_excel(data_path)
df = df.drop(["Recording Date", "Defect Code"], axis=1)

rf_model_path = r'models\with_delta_values\binary\NOTNORM_binary_random_forest_model.pkl'
xgb_model_path = r'models\with_delta_values\binary\NOTNORM_binary_xgb_model.json'
catboost_model_path = r'models\with_delta_values\binary\NOTNORM_binary_catboost_model.cbm'

# load models
rf_model = joblib.load(rf_model_path)
xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)
catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)


# function to obtain the defect score
def defect_score(x):

    x = x.reshape(1, -1)

    rf_prob = rf_model.predict_proba(x)
    xgb_prob = xgb_model.predict_proba(x)
    catboost_prob = catboost_model.predict_proba(x)

    avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

    return avg_defect_score


# # using MSE
# def fitness_function(x, target_defect_score, features_space):
#
#     x_concat = build_feature_array(x, features_space)
#     current_defect_score = defect_score(x_concat)
#
#     return mean_squared_error(current_defect_score, [target_defect_score])



# # # using log-cosh loss
# def fitness_function(x, target_defect_score, features_space):
#
#     x_concat = build_feature_array(x, features_space)
#     current_defect_score = defect_score(x_concat)
#
#     # log-cosh loss calculation -> log-cosh loss=log(cosh(predicted−actual))
#     delta = current_defect_score - target_defect_score
#     loss = np.log(np.cosh(delta))
#
#     return np.mean(loss)

# # using absolute error
def fitness_function(x, target_defect_score, features_space):
    x_concat = build_feature_array(x, features_space)
    current_defect_score = defect_score(x_concat)

    abs_diff = np.abs(current_defect_score - target_defect_score)

    return abs_diff
def build_feature_array(x, features_space):

    x_concat = np.zeros(len(features_space))
    x_list = list(x)
    for i, v in enumerate(features_space):
        if type(v[1]) != tuple:
            x_concat[i] = v[1]

        else:
            x_concat[i] = x_list.pop(0)
    return x_concat

# dual annealing callback function
def dual_annealing_callback(x, f, context):

    columns_list = df.columns.tolist()
    real_time_param = {feature: value.round(2) for feature, value in zip(columns_list, x) if
                       feature in ['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature',
                                   'Upper Plate Temperature']}
    print('\nReal-time adjustable params:', real_time_param)
    # print('MSE:', f.round(3))
    print('Log-Cosh Loss:', f.round(3))

# powell and nelder mead callback function
def minimize_callback(xk):
    print(xk)

def optimize_params(features_space, x0, target_defect_score, cb=dual_annealing_callback):

    for i, v in enumerate(features_space):
        if v[1] is None:
            features_space[i][1] = (df[v[0]].min(), df[v[0]].max())

    nff_idx, bounds = zip(*[(i, v[1]) for i, v in enumerate(features_space) if type(v[1]) == tuple])
    x0_filtered = [v for i, v in enumerate(x0) if i in set(nff_idx)]

    if dual_annealing_optim == 1:

        result = dual_annealing(
            func=fitness_function,
            x0=x0_filtered,
            bounds=bounds,
            args= [target_defect_score, features_space],
            callback=cb,
            maxfun=1e3,
            seed=16
        )

    if powell_optim == 1:

        result = minimize(fitness_function, x0_filtered,
                       method='Powell',
                       bounds=bounds,
                       callback=minimize_callback,
                       args=(target_defect_score, features_space),
                       options={'maxiter': 5000, 'disp': True},
                       tol=1e-6)

    if nelder_mead_optim == 1:

        result = minimize(fitness_function, x0_filtered,
                       method='Nelder-Mead',
                       bounds=bounds,
                       callback=minimize_callback,
                       args=(target_defect_score, features_space),
                       options={'maxiter': 5000, 'disp': True},
                       tol=1e-6)

    if basinhopping_optim == 1:

        minimizer_kwargs = {
            'method': 'BFGS',
            'bounds': bounds,
            'args': (target_defect_score, features_space),
            'options': {'maxiter': 5000, 'disp': True, 'tol': 1e-6}
        }

        result = basinhopping(
            func=fitness_function,
            x0=x0_filtered,
            minimizer_kwargs=minimizer_kwargs,
            # callback=minimize_callback,
            niter=10,
            disp=False
        )

    best_params = build_feature_array(result.x, features_space)
    mse = result.fun

    return best_params, mse


# good sample for reference (no defect)
x0 = df.iloc[20]

# sample with defect to optimize
sample = df.iloc[252]
sample_array = np.array(sample)
initial_defect_score = defect_score(sample_array)

features_space = []

# append to features space the features name and their respective value given the considered sample
for column in df.columns:
    features_space.append([column, sample[column]])

# intervals for the features that can be adjusted in real-time
intervals = {
    'Thermal Cycle Time': (10, 150),
    'Pressure': (250, 350),
    'Lower Plate Temperature': (160, 210),
    'Upper Plate Temperature': (160, 210)
}

# updates the values (bounds) for the real time features in the features_space
for feature, value in features_space:
    if feature in intervals:
        features_space[features_space.index([feature, value])][1] = intervals[feature]

# indices of the real time features in the features_space (will be used for printing some values)
thermal_cycle_time_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Thermal Cycle Time'][0]
pressure_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Pressure'][0]
lower_plate_temp_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Lower Plate Temperature'][0]
upper_plate_temp_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Upper Plate Temperature'][0]

indices = [thermal_cycle_time_index, pressure_index, lower_plate_temp_index, upper_plate_temp_index]

if dual_annealing_optim == 1:
    print("\nusing Dual Annealing...")
if powell_optim == 1:
    print("\nusing Powell...")
if nelder_mead_optim == 1:
    print("\nusing Nelder Mead...")
if basinhopping_optim == 1:
    print("\nusing Basin Hopping...")

print("\n---- Before Optimization ----")

sample_values = sample[['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature', 'Upper Plate Temperature']]
print("\nInitial parameter values: ", sample_values.values.round(2))

print(f"\nStarting optimization...")

target_defect_score = 0.5

best_params, mse = optimize_params(features_space, x0, target_defect_score)
final_defect_score = defect_score(best_params)

best_params_selected = best_params[indices]

# results
print('\n---- Optimization Results ----')
print('\nTarget Defect Score:   ', target_defect_score)
print('Best Parameters:    ', best_params_selected.round(2))
print('Initial Defect Score:  ', initial_defect_score)
print('Final Defect Score:    ', final_defect_score)
# print('MSE:                ', mse.round(3))
print('MSE:                ', mse)

