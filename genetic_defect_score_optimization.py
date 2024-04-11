import numpy as np
import pandas as pd
import joblib
import warnings
import time
import random
import pygad

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

print("Loaded Models!")

# function to obtain the defect score
def defect_score(x):

    x = x.reshape(1, -1)

    rf_prob = rf_model.predict_proba(x)
    xgb_prob = xgb_model.predict_proba(x)
    catboost_prob = catboost_model.predict_proba(x)

    avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

    return avg_defect_score


# using MSE
def fitness_function(ga_instance, solution, solution_idx):
    target_defect_score = 0.1  # Set your target defect score here
    current_defect_score = defect_score(solution)
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

def create_features_space(defect_sample_index):

    sample = df.iloc[defect_sample_index]
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

    bounds = []

    for i, v in enumerate(features_space):
        if v[1] is None:
            features_space[i][1] = (df[v[0]].min(), df[v[0]].max())
        bounds.append(v[1])

    return features_space, bounds, initial_defect_score


features_space, bounds, initial_defect_score = create_features_space(55230)
print(f'\nInitial Defect Score {initial_defect_score}')

# Define GA parameters
num_generations = 100
num_parents_mating = 10

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=num_parents_mating,
                       num_genes=len(features_space),
                       gene_space=list(bounds),
                       initial_population=None)

# Start optimization
ga_instance.run()

# Get the best solution and its fitness value
solution, solution_fitness = ga_instance.best_solution()

print("Best Solution:", solution)
print("Best Fitness:", solution_fitness)

#
#
#
