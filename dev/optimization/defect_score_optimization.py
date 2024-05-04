import numpy as np
import pandas as pd
import joblib
import warnings
import time
import random

from scipy.optimize import dual_annealing, minimize, basinhopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Test data
data_path = r'../data/clean_data/binary_cleaned_data_2022_line410.xlsx'
df = pd.read_excel(data_path)
df = df.drop(["Recording Date", "Defect Code", "Group"], axis=1)

# Load Prediction Models
rf_model_path = r'../prediction/models/binary/binary_random_forest_model.pkl'
xgb_model_path = r'../prediction/models/binary/binary_xgb_model.json'
catboost_model_path = r'../prediction/models/binary/binary_catboost_model.cbm'

rf_model = joblib.load(rf_model_path)
xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)
catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)

# Select the optimization method
dual_annealing_optim = 1
powell_optim = 0
nelder_mead_optim = 0
basinhopping_optim = 0

method_name = ''
df_name = ''

if dual_annealing_optim == 1:
    method_name = 'dual_annealing'
if powell_optim == 1:
    method_name = 'powell'
if nelder_mead_optim == 1:
    method_name = 'nelder_mead'
if basinhopping_optim == 1:
    method_name = 'basinhopping'

# Use random indexes for the sample -> 1
# Use the same sample indexes (for consistency when comparing results) -> 0
random_index = 0

# To store the average optimization result for each defect score range
average_results = []

# Calculate the defect score for the all sample to be separate into the different ranges
rf_prob = rf_model.predict_proba(df)
xgb_prob = xgb_model.predict_proba(df)
catboost_prob = catboost_model.predict_proba(df)

avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)
df["Defect Score"] = avg_defect_score

# Separate the sample with different defect score ranges
df_01_03 = df[(df['Defect Score'] >= 0.1) & (df['Defect Score'] < 0.3)]
df_03_05 = df[(df['Defect Score'] >= 0.3) & (df['Defect Score'] < 0.5)]
df_05_06 = df[(df['Defect Score'] >= 0.5) & (df['Defect Score'] < 0.6)]
df_06_07 = df[(df['Defect Score'] >= 0.6) & (df['Defect Score'] < 0.7)]
df_07_08 = df[(df['Defect Score'] >= 0.7) & (df['Defect Score'] < 0.8)]
df_08_09 = df[(df['Defect Score'] >= 0.8) & (df['Defect Score'] < 0.9)]
df_09_95 = df[(df['Defect Score'] >= 0.9) & (df['Defect Score'] < 0.95)]
df_95_99 = df[(df['Defect Score'] >= 0.95) & (df['Defect Score'] < 0.99)]
df_99_1 = df[(df['Defect Score'] >= 0.99) & (df['Defect Score'] <= 1)]

all_dfs = [df_01_03, df_03_05, df_05_06, df_06_07, df_07_08, df_08_09, df_09_95, df_95_99, df_99_1]

# df = df.drop(["Defect Score"], axis=1)

# Run the optimization for each defect score range
for test_df in all_dfs:

    # Variables to store the avg. results
    total_initial_defect_score = 0
    total_final_defect_score = 0
    total_elapsed_time = 0
    total_reduction = 0

    count_0 = 0
    count_01 = 0
    count_05 = 0

    random_defect_indexes = None

    # Get the current df name
    if test_df.equals(df_01_03):
        df_name = "df_01_03"
    elif test_df.equals(df_03_05):
        df_name = "df_03_05"
    elif test_df.equals(df_05_06):
        df_name = "df_05_06"
    elif test_df.equals(df_06_07):
        df_name = "df_06_07"
    elif test_df.equals(df_07_08):
        df_name = "df_07_08"
    elif test_df.equals(df_08_09):
        df_name = "df_08_09"
    elif test_df.equals(df_09_95):
        df_name = "df_09_95"
    elif test_df.equals(df_95_99):
        df_name = "df_95_99"
    elif test_df.equals(df_99_1):
        df_name = "df_99_1"

    # Select 50 random samples and save their indexes in a file
    if random_index == 1:

        sample_size = min(50, len(test_df))
        random_defect_indexes = np.random.choice(test_df.index, size=sample_size, replace=False)

        random_index_df = pd.DataFrame({'DF': [df_name], 'Indexes': [','.join(map(str, random_defect_indexes))]})
        random_index_df.to_csv('optimization_df_indexes.csv', mode='a', header=False, index=False)


    # Use the indexes already stored in the file (for consistency when comparing results between methods - always using
    # the same samples)
    elif random_index == 0:

        with open('optimization_df_indexes.csv', 'r') as file:

            for line in file:

                row_data = line.strip().replace('"', '').replace("'", "").split(',')
                current_df_name = row_data[0]

                if current_df_name == df_name:

                    random_defect_indexes = row_data[1:]
                    random_defect_indexes = [int(index) for index in random_defect_indexes]
                    break


    # Function to get defect score
    def defect_score(x):

        x = x.reshape(1, -1)

        rf_prob = rf_model.predict_proba(x)
        xgb_prob = xgb_model.predict_proba(x)
        catboost_prob = catboost_model.predict_proba(x)

        avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

        return avg_defect_score


    # Fitness Functions

    # Using MSE
    def fitness_function(x, target_defect_score, features_space):

        x_concat = build_feature_array(x, features_space)
        current_defect_score = defect_score(x_concat)

        return mean_squared_error(current_defect_score, [target_defect_score])


    # # MSE without any target score
    # def fitness_function(x, target_defect_score, features_space):
    #
    #     x_concat = build_feature_array(x, features_space)
    #     current_defect_score = defect_score(x_concat)
    #
    #     return mean_squared_error(current_defect_score, [current_defect_score])

    # # Using Log-Cosh Loss
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

    # # Using Absolute Error ([M]AE)
    # def fitness_function(x, target_defect_score, features_space):
    #
    #     x_concat = build_feature_array(x, features_space)
    #     current_defect_score = defect_score(x_concat)
    #
    #     abs_diff = np.abs(current_defect_score - target_defect_score)
    #
    #     return abs_diff

    # Function that concats the four real-time features with the rest of the sample features
    def build_feature_array(x, features_space):

        x_concat = np.zeros(len(features_space))
        x_list = list(x)

        for i, v in enumerate(features_space):

            if type(v[1]) != tuple:
                x_concat[i] = v[1]

            else:
                x_concat[i] = x_list.pop(0)

        return x_concat

    # Optimization function
    def optimize_params(features_space, x0, target_defect_score):

        nff_idx, bounds = zip(*[(i, v[1]) for i, v in enumerate(features_space) if type(v[1]) == tuple])
        x0_filtered = [v for i, v in enumerate(x0) if i in set(nff_idx)]

        # Dual Anneling
        if dual_annealing_optim == 1:

            result = dual_annealing(
                func=fitness_function,
                x0=x0_filtered,
                bounds=bounds,
                args=[target_defect_score, features_space],
                maxfun=1e3,
                seed=16
            )

        # Powell
        if powell_optim == 1:

            result = minimize(fitness_function, x0_filtered,
                              method='Powell',
                              bounds=bounds,
                              args=(target_defect_score, features_space),
                              options={'maxiter': 5000, 'disp': False},
                              tol=1e-6)

        # Nelder Mead
        if nelder_mead_optim == 1:

            result = minimize(fitness_function, x0_filtered,
                              method='Nelder-Mead',
                              bounds=bounds,
                              args=(target_defect_score, features_space),
                              options={'maxiter': 5000, 'disp': False},
                              tol=1e-6)
        # Basin Hopping
        if basinhopping_optim == 1:

            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'bounds': bounds,
                'args': (target_defect_score, features_space),
                'options': {'maxiter': 5000, 'disp': False, 'tol': 1e-6}
            }

            result = basinhopping(
                func=fitness_function,
                x0=x0_filtered,
                minimizer_kwargs=minimizer_kwargs,
                niter=10,
                disp=False
            )

        best_params = build_feature_array(result.x, features_space)
        mse = result.fun

        return best_params, mse


    # Good sample for reference (with very small defect score)
    x0 = df.iloc[20]

    # To store results
    score_reducing = []
    time_spent = []

    if dual_annealing_optim == 1:
        print("\nusing Dual Annealing...")
    if powell_optim == 1:
        print("\nusing Powell...")
    if nelder_mead_optim == 1:
        print("\nusing Nelder Mead...")
    if basinhopping_optim == 1:
        print("\nusing Basin Hopping...")


    # Loop through every sample
    for defect_sample in random_defect_indexes:

        sample = df.iloc[defect_sample]
        sample_array = np.array(sample)

        # Get the defect score before the optimization
        initial_defect_score = defect_score(sample_array)

        # Append to features space the features name and their respective value given the considered sample

        features_space = []

        for column in df.columns:
            features_space.append([column, sample[column]])

        # Intervals for the features that can be adjusted in real-time
        intervals = {
            'Thermal Cycle Time': (10, 150),
            'Pressure': (250, 350),
            'Lower Plate Temperature': (160, 210),
            'Upper Plate Temperature': (160, 210)
        }

        # Updates the values (bounds) for the real time features in the features_space
        for feature, value in features_space:

            if feature in intervals:
                features_space[features_space.index([feature, value])][1] = intervals[feature]

        # Indices of the real time features in the features_space
        thermal_cycle_time_index = \
            [i for i, (feature, _) in enumerate(features_space) if feature == 'Thermal Cycle Time'][0]
        pressure_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Pressure'][0]
        lower_plate_temp_index = \
            [i for i, (feature, _) in enumerate(features_space) if feature == 'Lower Plate Temperature'][0]
        upper_plate_temp_index = \
            [i for i, (feature, _) in enumerate(features_space) if feature == 'Upper Plate Temperature'][0]

        indices = [thermal_cycle_time_index, pressure_index, lower_plate_temp_index, upper_plate_temp_index]

        # initial_parameters = [sample[0][index] for index in indices]

        sample_values = sample[['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature', 'Upper Plate Temperature']]


        # Initialize variables to store the best optimization results between the target score
        best_reduction_percentage = -float('inf')
        best_target_defect_score = None
        best_mse = None
        best_final_defect_score = None
        best_params_selected = None
        best_elapsed_time = None

        # Defining the target score for the fitness function
        target_defect_scores = [0, 0.1, 0.5]

        # Loop through each target defect score
        for target_defect_score in target_defect_scores:

            # Start timer
            start_time = time.time()

            # Optimize parameters for the current target defect score
            current_params, current_mse = optimize_params(features_space, x0, target_defect_score)
            current_final_defect_score = defect_score(current_params)
            current_reduction_percentage = (initial_defect_score - current_final_defect_score) * 100

            # Update best result if the current reduction percentage is higher
            if current_reduction_percentage > best_reduction_percentage:
                best_reduction_percentage = current_reduction_percentage
                best_target_defect_score = target_defect_score
                best_mse = current_mse
                best_final_defect_score = current_final_defect_score
                best_params_selected = current_params[indices]
                best_elapsed_time = time.time() - start_time

        # Print the best optimization results
        print('\n**** Best Optimization Results ****')
        print('Target Defect Score:   ', best_target_defect_score)
        # print('Initial Parameters:    ', initial_parameters)
        print('Best Parameters:    ', best_params_selected.round(2))
        print('Initial Defect Score:  ', initial_defect_score)
        print('Final Defect Score:    ', best_final_defect_score)
        print('Reduced Defect Score in:    ', best_reduction_percentage, '%')
        print('Elapsed Time (in seconds):    ', round(best_elapsed_time, 2))
        print('MSE:                ', best_mse.round(3))

        # Update total values for averages with the best result from the three target scores
        total_initial_defect_score += initial_defect_score
        total_final_defect_score += best_final_defect_score
        total_elapsed_time += best_elapsed_time
        total_reduction += best_reduction_percentage

        # Counts the number of times that each target defect score achieved the best optim result
        if best_target_defect_score == 0:
            count_0 = count_0 + 1
        if best_target_defect_score == 0.1:
            count_01 = count_01 + 1
        if best_target_defect_score == 0.5:
            count_05 = count_05 + 1

    # Calculate the average defect score before and after optimization, average time spent, and average reduction
    average_initial_defect_score = total_initial_defect_score / len(random_defect_indexes)
    average_final_defect_score = total_final_defect_score / len(random_defect_indexes)
    average_elapsed_time = total_elapsed_time / len(random_defect_indexes)
    average_reduction = total_reduction / len(random_defect_indexes)

    # Print the average results for each defect score range
    print(f"\n---------- AVERAGE RESULTS {df_name}----------")

    print(f'Average Defect Score before Optimization: {average_initial_defect_score}')
    print(f'Average Defect Score after Optimization: {average_final_defect_score}')
    print(f'Average Time Spent (in seconds): {round(average_elapsed_time, 2)}')
    print(f'Average Reduction in Defect Score: {average_reduction}%')

#     # To save the results to a file
#
#     average_results.append({
#         'DF': df_name,
#         'Avg Defect Score Before Optimization': average_initial_defect_score,
#         'Avg Defect Score After Optimization': average_final_defect_score,
#         'Avg Time Spent': round(average_elapsed_time, 2),
#         'Avg Reduction': average_reduction,
#         'Target Score 0': count_0,
#         'Target Score 0.1': count_01,
#         'Target Score 0.5': count_05,
#     })
#
# average_results_df = pd.DataFrame(average_results)
# average_results_df.to_excel('wts_' + method_name + '_optimization_average_results.xlsx', index=False)
