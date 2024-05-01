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

dual_annealing_optim = 1
powell_optim = 0
nelder_mead_optim = 0
basinhopping_optim = 0
warnings.filterwarnings("ignore")

data_path = r'../data/clean_data/binary_cleaned_data_2022_line410.xlsx'

df = pd.read_excel(data_path)

# defect_rows = df[df['Defect Code'] == 1]
# random_defect_indexes = np.random.choice(defect_rows.index, size=200, replace=False)

df = df.drop(["Recording Date", "Defect Code", "Group"], axis=1)

rf_model_path = r'../models/without_delta_values/binary/binary_random_forest_model.pkl'
xgb_model_path = r'../models/without_delta_values/binary/binary_xgb_model.json'
catboost_model_path = r'../models/without_delta_values/binary/binary_catboost_model.cbm'

# load models
rf_model = joblib.load(rf_model_path)
xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)
catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)

# defect socre for the whole df to later separte to test
rf_prob = rf_model.predict_proba(df)
xgb_prob = xgb_model.predict_proba(df)
catboost_prob = catboost_model.predict_proba(df)

avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)
df["Defect Score"] = avg_defect_score

df_01_03 = df[(df['Defect Score'] >= 0.1) & (df['Defect Score'] < 0.3)]
df_03_05 = df[(df['Defect Score'] >= 0.3) & (df['Defect Score'] < 0.5)]
df_05_06 = df[(df['Defect Score'] >= 0.5) & (df['Defect Score'] < 0.6)]
df_06_07 = df[(df['Defect Score'] >= 0.6) & (df['Defect Score'] < 0.7)]
df_07_08 = df[(df['Defect Score'] >= 0.7) & (df['Defect Score'] < 0.8)]
df_08_09 = df[(df['Defect Score'] >= 0.8) & (df['Defect Score'] < 0.9)]
df_09_95 = df[(df['Defect Score'] >= 0.9) & (df['Defect Score'] < 0.95)]
df_95_99 = df[(df['Defect Score'] >= 0.95) & (df['Defect Score'] < 0.99)]
df_99_1 = df[(df['Defect Score'] >= 0.99) & (df['Defect Score'] <= 1)]

all_dfs = [ df_07_08]

# all_dfs = [df_01_03, df_03_05, df_05_06, df_06_07, df_07_08, df_08_09, df_09_95, df_95_99, df_99_1]

for test_df in all_dfs:
    print(len(test_df))


df = df.drop(["Defect Score"], axis=1)

for test_df in all_dfs:

    sample_size = min(10, len(test_df))  # select 10 samples or all samples if less than 10
    random_defect_indexes = np.random.choice(test_df.index, size=sample_size, replace=False)

    # random_defect_indexes = np.random.choice(test_df.index, size=15, replace=False)

    # function to obtain the defect score
    def defect_score(x):

        x = x.reshape(1, -1)

        rf_prob = rf_model.predict_proba(x)
        xgb_prob = xgb_model.predict_proba(x)
        catboost_prob = catboost_model.predict_proba(x)

        avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

        return avg_defect_score


    # # using MSE
    def fitness_function(x, target_defect_score, features_space):

        x_concat = build_feature_array(x, features_space)
        current_defect_score = defect_score(x_concat)

        return mean_squared_error(current_defect_score, [target_defect_score])


    # TESTE E SEM TER EM CONTA O TARGET DEFECT SCORE
    # def fitness_function(x, target_defect_score, features_space):
    #     x_concat = build_feature_array(x, features_space)
    #     current_defect_score = defect_score(x_concat)
    #     # Return the mean squared error of the current defect score
    #     # with respect to itself, effectively minimizing the defect score
    #     return mean_squared_error(current_defect_score, [current_defect_score])

    # # using log-cosh loss
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

    # # using absolute error ([M]AE)
    # def fitness_function(x, target_defect_score, features_space):
    #     x_concat = build_feature_array(x, features_space)
    #     current_defect_score = defect_score(x_concat)
    #
    #     abs_diff = np.abs(current_defect_score - target_defect_score)
    #
    #     return abs_diff

    def build_feature_array(x, features_space):

        print(x)

        x_concat = np.zeros(len(features_space))
        x_list = list(x)
        for i, v in enumerate(features_space):
            if type(v[1]) != tuple:
                x_concat[i] = v[1]

            else:
                x_concat[i] = x_list.pop(0)

        print(x_concat)
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
    # def minimize_callback(xk):
    #     # print(xk)

    def optimize_params(features_space, x0, target_defect_score, t, cb=dual_annealing_callback):


        print(t)
        print(features_space)

        if t == 'com':
            for i, v in enumerate(features_space):
                if v[1] is None:
                    features_space[i][1] = (df[v[0]].min(), df[v[0]].max())

        nff_idx, bounds = zip(*[(i, v[1]) for i, v in enumerate(features_space) if type(v[1]) == tuple])
        x0_filtered = [v for i, v in enumerate(x0) if i in set(nff_idx)]

        print(bounds)

        if dual_annealing_optim == 1:
            result = dual_annealing(
                func=fitness_function,
                x0=x0_filtered,
                bounds=bounds,
                args=[target_defect_score, features_space],
                # callback=cb,
                maxfun=1e3,
                seed=16
            )

        if powell_optim == 1:
            result = minimize(fitness_function, x0_filtered,
                              method='Powell',
                              bounds=bounds,
                              # callback=minimize_callback,
                              args=(target_defect_score, features_space),
                              options={'maxiter': 5000, 'disp': False},
                              tol=1e-6)

        if nelder_mead_optim == 1:
            result = minimize(fitness_function, x0_filtered,
                              method='Nelder-Mead',
                              bounds=bounds,
                              # callback=minimize_callback,
                              args=(target_defect_score, features_space),
                              options={'maxiter': 5000, 'disp': False},
                              tol=1e-6)

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
                # callback=minimize_callback,
                niter=10,
                disp=False
            )

        best_params = build_feature_array(result.x, features_space)
        mse = result.fun

        return best_params, mse


    # good sample for reference (no defect)
    x0 = df.iloc[20]

    if test_df.equals(df_01_03):
        defect_samples = [5265, 13891, 11970, 11875, 6016, 5196, 11208, 14334, 4980, 2890]
    elif test_df.equals(df_03_05):
        defect_samples = [14057, 5784, 2210, 684, 5249, 9865, 1368, 6857, 7502, 15514]
    elif test_df.equals(df_05_06):
        defect_samples = [9337, 12767, 1509, 9870, 4262, 9330, 6865, 13605, 5187, 8750]
    elif test_df.equals(df_06_07):
        defect_samples = [6006, 13581, 2280, 10465, 9886, 10118, 6210, 6007, 11940, 6008]
    elif test_df.equals(df_07_08):
        defect_samples = [11958, 4925, 7388, 3837, 281, 10977, 14734, 4591, 6121, 11938]
    elif test_df.equals(df_08_09):
        defect_samples = [1822, 13010, 14464, 2018, 9756, 12, 1725, 2899, 3004, 529]
    elif test_df.equals(df_09_95):
        defect_samples = [764, 11071, 1269, 12229, 3015, 9009, 8552, 13979, 7299, 8679]
    elif test_df.equals(df_95_99):
        defect_samples = [15351, 8819, 9650, 8771, 14936, 5838, 15343, 7511, 7725, 8561]
    elif test_df.equals(df_99_1):
        defect_samples = [6790, 14359, 14375, 11409, 15656, 2153, 14376, 7026, 586, 11900]

    # defect_samples = random_defect_indexes

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

    for defect_sample in defect_samples:

        sample = df.iloc[defect_sample]
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
        thermal_cycle_time_index = \
        [i for i, (feature, _) in enumerate(features_space) if feature == 'Thermal Cycle Time'][0]
        pressure_index = [i for i, (feature, _) in enumerate(features_space) if feature == 'Pressure'][0]
        lower_plate_temp_index = \
        [i for i, (feature, _) in enumerate(features_space) if feature == 'Lower Plate Temperature'][0]
        upper_plate_temp_index = \
        [i for i, (feature, _) in enumerate(features_space) if feature == 'Upper Plate Temperature'][0]

        indices = [thermal_cycle_time_index, pressure_index, lower_plate_temp_index, upper_plate_temp_index]

        # print("\n---- Before Optimization ----")

        sample_values = sample[['Thermal Cycle Time', 'Pressure', 'Lower Plate Temperature', 'Upper Plate Temperature']]
        # print("\nInitial parameter values: ", sample_values.values.round(2))
        #
        # print(f"\nStarting optimization...")

        # target_defect_scores = [0, 0.1, 0.5]
        target_defect_scores = [0.5]

        def_aux_01 = 0
        time_aux_01 = 0

        def_aux_05 = 0
        time_aux_05 = 0

        for target_defect_score in target_defect_scores:

            print('\n**********************************************')

            # target_defect_score = 0.5

            if test_df.equals(df_01_03) or test_df.equals(df_03_05):
                target_defect_score = 0

            # to count the time that the optimization took (in seconds)

            teste = ['com', 'sem']

            for t in teste:

                print('\n-------------------------------------------------')

                start_time = time.time()

                best_params, mse = optimize_params(features_space, x0, target_defect_score, t)

                end_time = time.time()
                elapsed_time = end_time - start_time

                final_defect_score = defect_score(best_params)

                best_params_selected = best_params[indices]

                reduction_percentage = (initial_defect_score - final_defect_score) * 100

                if target_defect_score == 0.1:
                    def_aux_01 = reduction_percentage
                    time_aux_01 = elapsed_time
                else:
                    def_aux_05 = reduction_percentage
                    time_aux_05 = elapsed_time

                score_reducing.append(reduction_percentage)
                time_spent.append(elapsed_time)

                # results
                print('\n---- Optimization Results ----')
                print('\nTarget Defect Score:   ', target_defect_score)
                # print('Best Parameters:    ', best_params_selected.round(2))
                print('Initial Defect Score:  ', initial_defect_score)
                print('Final Defect Score:    ', final_defect_score)
                print(f'Reduced Defect Score in {reduction_percentage}%')
                # print('Elapsed Time (in seconds):    ', round(elapsed_time, 2))
                # print('MSE:                ', mse.round(3))

        if def_aux_01 < def_aux_05:
            score_reducing.append(def_aux_01)
            time_spent.append(time_aux_01)

        else:
            score_reducing.append(def_aux_05)
            time_spent.append(time_aux_05)

        # if test_df.equals(df_01_03) or test_df.equals(df_03_05):
        #     break

    # avg results

    average_reduction_percentage = sum(score_reducing) / len(score_reducing)
    average_elapsed_time = sum(time_spent) / len(time_spent)

    print("\n---------- AVERAGE RESULTS ----------")

    if test_df.equals(df_01_03):
        print("df_01_03")
    elif test_df.equals(df_03_05):
        print("df_03_05")
    elif test_df.equals(df_05_06):
        print("df_05_06")
    elif test_df.equals(df_06_07):
        print("df_06_07")
    elif test_df.equals(df_07_08):
        print("df_07_08")
    elif test_df.equals(df_08_09):
        print("df_08_09")
    elif test_df.equals(df_09_95):
        print("df_09_95")
    elif test_df.equals(df_09_95):
        print("df_05_99")
    elif test_df.equals(df_99_1):
        print("df_99_1")

    # print(f'Target Defect Score: {target_defect_score}')
    print(f'Reduced Defect Score in {average_reduction_percentage}%')
    print('Elapsed Time (in seconds):    ', round(average_elapsed_time, 2))
    print('Indexes used:' , random_defect_indexes)