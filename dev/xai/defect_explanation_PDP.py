import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# Load train and test data
x_train = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_x_train_aug.xlsx')
y_train = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_y_train_aug.xlsx')
x_test = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_x_test.xlsx')
y_test = pd.read_excel(
    r'..\data\split_train_test_data\binary_data\binary_y_test.xlsx')

print("Loaded data!")

catboost_model_path = r'../prediction/models/binary/binary_catboost_model.cbm'
catboost_model = CatBoostClassifier()
catboost_model.load_model(catboost_model_path)

predictions = catboost_model.predict_proba(x_test)[:, 1]

features = ['Width', 'Length', 'Press Input Table Speed', 'Lower Plate Temperature']

pdp_display = PartialDependenceDisplay.from_estimator(catboost_model, x_test, features, kind='both', centered=True, response_method='predict_proba')
fig, ax = plt.subplots(figsize=(40, 10))
pdp_display.plot(ax=ax)
plt.show()
