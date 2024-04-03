import numpy as np
import pandas as pd
import joblib
import warnings

from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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


def fitness_function(x, target_defect_score):
    current_defect_score = defect_score(x)
    # print(f'current_defect_score: {current_defect_score}')
    return mean_squared_error([target_defect_score], [current_defect_score])


# callback function
def dual_annealing_callback(x, f, context):
    real_time_param = {feature: value.round(2) for feature, value in zip(df.columns, x) if
                       feature in ['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature',
                                   'Upper Plate Temperature']}
    print('\nReal-time adjustable params:', real_time_param)
    print('Defect score:', f.round(3))


def optimize_params(features_space, x0, target_defect_score, cb=dual_annealing_callback):
    bounds = [bounds_tuple[1] for bounds_tuple in features_space if isinstance(bounds_tuple[1], tuple)]

    result = dual_annealing(
        func=fitness_function,
        x0=x0,
        bounds=bounds,
        # args= [target_defect_score, features_space],  # depois confirmar se assim está correto
        args=(target_defect_score,),
        callback=cb,
        maxfun=1e3,
        seed=16
    )

    best_params = result.x
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

# since the function requires a bound even though i want to keep some values "fixed",
# i'm subtracting and adding a small value around the value to it creates a valid interval
# without affecting

aux = 0.0001

for i, (feature, value) in enumerate(features_space):

    if feature not in intervals:
        bounds_min = max(0, value - aux)  # ensuring that the bounds are not negative
        bounds_max = max(0, value + aux)
        features_space[i][1] = (bounds_min, bounds_max)

# indices of the real time features in the features_space (will be used for printing some values)
thermal_cycle_time_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Thermal Cycle Time'][0]
pressure_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Pressure'][0]
lower_plate_temp_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Lower Plate Temperature'][0]
upper_plate_temp_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Upper Plate Temperature'][0]

indices = [thermal_cycle_time_index, pressure_index, lower_plate_temp_index, upper_plate_temp_index]

print("\n---- Before Optimization ----")
sample_values = df.iloc[252][['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature', 'Upper Plate Temperature']]
# print(sample_values)
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
print('MSE:                ', mse.round(3))

# best_params = best_params.reshape(1, -1)
# rf_prob = rf_model.predict_proba(best_params)
# xgb_prob = xgb_model.predict_proba(best_params)
# catboost_prob = catboost_model.predict_proba(best_params)
# print("\nProbs")
# print(rf_prob[:, 1])
# print(xgb_prob[:, 1])
# print(catboost_prob[:, 1])

# compare params before and after optim to see if it changed any feature value besides the real-time ones

# # best_parms --> numpy.ndarray
# # sample --> class 'pandas.core.series.Series

# sample_v = sample.values
# num_elements = len(best_params[0])
#
# for i in range(num_elements):
#     if best_params[0][i] == sample_v[i]:
#         print(f"Index {i}: Values are equal - Best Params: {best_params[0][i]}, Sample: {sample_v[i]}")
#     else:
#         print(f"Index {i}: Values are not equal - Best Params: {best_params[0][i]}, Sample: {sample_v[i]}")



