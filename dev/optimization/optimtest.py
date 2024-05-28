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


def defect_score(x):

    x = x.reshape(1, -1)

    rf_prob = rf_model.predict_proba(x)
    xgb_prob = xgb_model.predict_proba(x)
    catboost_prob = catboost_model.predict_proba(x)

    avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

    return avg_defect_score


# Sample Before
sample_before = df.iloc[13603]
# print('Sample Before Optimization:')
# print(sample_before)

defect_score_before = defect_score(sample_before.values)
print(f'Sample Score Before Optimization:' + {defect_score_before[0]*100} + '%')


# After Optim
sample_after = df.iloc[13603].copy()

sample_after['Thermal Cycle Time'] = 24
sample_after['Pressure'] = 280
sample_after['Lower Plate Temperature'] = 169
sample_after['Upper Plate Temperature'] = 191
sample_after['Cycle Time'] = 38
sample_after['Mechanical Cycle Time'] = 14.8
sample_after['Carriage Speed'] = 1400
sample_after['Press Input Table Speed'] = 1200
sample_after['Scraping Cycle'] = 65521.0
sample_after['Transverse Saw Cycle'] = 0.0

# print('Sample After Optimization:')
# print(sample_after)

defect_score_after = defect_score(sample_after.values)
print(f'Sample Score After Optimization:' + {defect_score_after[0]*100} + '%')
