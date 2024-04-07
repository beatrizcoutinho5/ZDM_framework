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

dual_annealing_optim = 0
powell_optim = 0
nelder_mead_optim = 0
basinhopping_optim = 1

warnings.filterwarnings("ignore")

data_path = r'data\clean_data\binary_cleaned_data_with_deltavalues_2022_2023_2024.xlsx'

df = pd.read_excel(data_path)

# defect_rows = df[df['Defect Code'] == 1]
# random_defect_indexes = np.random.choice(defect_rows.index, size=200, replace=False)

df = df.drop(["Recording Date", "Defect Code", "Group"], axis=1)

rf_model_path = r'models\with_delta_values\binary\binary_random_forest_model.pkl'
xgb_model_path = r'models\with_delta_values\binary\binary_xgb_model.json'
catboost_model_path = r'models\with_delta_values\binary\binary_catboost_model.cbm'

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

all_dfs = [df_01_03, df_03_05, df_05_06, df_06_07, df_07_08, df_08_09, df_09_95, df_95_99, df_99_1]

df = df.drop(["Defect Score"], axis=1)
for test_df in all_dfs:

    # random_defect_indexes = np.random.choice(test_df.index, size=15, replace=False)

    # function to obtain the defect score
    def defect_score(x):

        x = x.reshape(1, -1)

        rf_prob = rf_model.predict_proba(x)
        xgb_prob = xgb_model.predict_proba(x)
        catboost_prob = catboost_model.predict_proba(x)

        avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

        return avg_defect_score


    # using MSE
    def fitness_function(x, target_defect_score, features_space):

        x_concat = build_feature_array(x, features_space)
        current_defect_score = defect_score(x_concat)

        return mean_squared_error(current_defect_score, [target_defect_score])


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
    # def fitness_function(x, target_defect_score, features_space):
    #     x_concat = build_feature_array(x, features_space)
    #     current_defect_score = defect_score(x_concat)
    #
    #     abs_diff = np.abs(current_defect_score - target_defect_score)
    #
    #     return abs_diff
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
    # def minimize_callback(xk):
    #     # print(xk)

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
        defect_samples = [55138, 54832, 55586, 1409, 55200, 55625, 54382, 55960, 55018, 52932, 8999, 52941, 55771,
                          54813, 54841]
    elif test_df.equals(df_03_05):
        defect_samples = [43470, 9440, 43626, 55860, 53332, 54940, 55763, 54219, 9438, 37059, 43990, 54021, 54152,
                          54389, 53467]
    elif test_df.equals(df_05_06):
        defect_samples = [4924, 55667, 6776, 55023, 46497, 53901, 56013, 53409, 53922, 53926, 53689, 44107, 55295,
                          55490, 27021]
    elif test_df.equals(df_06_07):
        defect_samples = [55227, 54100, 29445, 52827, 54282, 53228, 7643, 53154, 54613, 54101, 9455, 51288, 45778,
                          54759, 38823]
    elif test_df.equals(df_07_08):
        defect_samples = [27012, 25691, 4494, 43217, 53153, 55230, 55965, 52705, 54542, 53870, 55238, 45865, 46934,
                          53124, 43464]
    elif test_df.equals(df_08_09):
        defect_samples = [933, 6703, 39713, 53176, 27015, 46084, 54548, 36256, 49148, 48144, 54103, 53030, 29594, 43416,
                          38857]
    elif test_df.equals(df_09_95):
        defect_samples = [52845, 39858, 18009, 55953, 29581, 8635, 22326, 27034, 20074, 51292, 23110, 55938, 29580,
                          23743, 45938]
    elif test_df.equals(df_95_99):
        defect_samples = [4749, 49861, 19973, 36825, 18143, 41954, 47214, 20166, 26509, 29604, 931, 40529,
                          52661, 28441, 1019]

    elif test_df.equals(df_99_1):
        defect_samples = [31189, 13006, 11701, 5623, 47555, 30668, 51754, 40313, 40102, 1272, 20468, 1770, 24774, 47248,
                          176]

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

        target_defect_score = 0.5

        if test_df.equals(df_01_03) or test_df.equals(df_03_05):
            target_defect_score = 0.1

        # to count the time that the optimization took (in seconds)
        start_time = time.time()

        best_params, mse = optimize_params(features_space, x0, target_defect_score)

        end_time = time.time()
        elapsed_time = end_time - start_time

        final_defect_score = defect_score(best_params)

        best_params_selected = best_params[indices]

        reduction_percentage = (initial_defect_score - final_defect_score) * 100

        score_reducing.append(reduction_percentage)
        time_spent.append(elapsed_time)

        # # results
        # print('\n---- Optimization Results ----')
        # print('\nTarget Defect Score:   ', target_defect_score)
        # print('Best Parameters:    ', best_params_selected.round(2))
        # print('Initial Defect Score:  ', initial_defect_score)
        # print('Final Defect Score:    ', final_defect_score)
        # print(f'Reduced Defect Score in {reduction_percentage}%')
        # print('Elapsed Time (in seconds):    ', round(elapsed_time, 2))
        # print('MSE:                ', mse.round(3))

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

    print(f'Target Defect Score: {target_defect_score}')
    print(f'Reduced Defect Score in {average_reduction_percentage}%')
    print('Elapsed Time (in seconds):    ', round(average_elapsed_time, 2))
