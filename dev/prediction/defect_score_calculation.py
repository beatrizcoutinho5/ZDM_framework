import numpy as np
import pandas as pd
import joblib
import warnings
import time

from scipy.optimize import dual_annealing, minimize, basinhopping
from scipy.spatial.distance import minkowski
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

data_path = r'../data/clean_data/binary_cleaned_data_2022_line410.xlsx'
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_excel(data_path)
df = df.drop(["Recording Date", "Group", "Defect Code"], axis=1)

rf_model_path = r'../models/without_delta_values/binary/binary_random_forest_model.pkl'
xgb_model_path = r'../models/without_delta_values/binary/binary_xgb_model.json'
catboost_model_path = r'../models/without_delta_values/binary/binary_catboost_model.cbm'

rf_model = joblib.load(rf_model_path)
xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)
catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)

rf_prob = rf_model.predict_proba(df)
xgb_prob = xgb_model.predict_proba(df)
catboost_prob = catboost_model.predict_proba(df)

avg_defect_score = np.mean([rf_prob[:, 1], xgb_prob[:, 1], catboost_prob[:, 1]], axis=0)

df["Defect Score"] = avg_defect_score
#
# df["Defect Code"] = pd.read_excel(data_path)["Defect Code"]
#
# # Rearrange columns to have "Defect Score" and "Defect Code" as the first columns
# cols = df.columns.tolist()
# cols = ['Defect Score', 'Defect Code'] + [col for col in cols if col not in ['Defect Score', 'Defect Code']]
# df = df[cols]

df["Recording Date"] = pd.read_excel(data_path)["Recording Date"]
df.to_excel(r'dev\data\split_train_test_data\defect_score_binary_cleaned_data_2022_line410.xlsx', index=False)
