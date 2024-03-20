import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier

# Suppress InconsistentVersionWarning
warnings.filterwarnings("ignore", category=UserWarning)

# For the Partial Dependence Plots (PDP) I will use the binary classification model (just predicts if it's a defect or
# not, it doesn't predict the defect type) to see which feature values affect the defect probability

# Load train and test data
x_train = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_x_train_aug.xlsx')
y_train = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_y_train_aug.xlsx')
x_test = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_x_test.xlsx')
y_test = pd.read_excel(
    r'data\split_train_test_data\with_delta_values\binary_data\NOTNORM_binary_y_test.xlsx')

print("Loaded data!")

# test with a regressor

model = XGBRegressor(verbose=True)

model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("True Values vs Predictions:")
for true_val, pred in zip(y_test.values.flatten(), predictions):
    print("True:", true_val, "\tPredicted:", pred)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

features = ['Length', 'Press Input Table Speed', 'Lower Plate Temperature']

pdp_display = PartialDependenceDisplay.from_estimator(model, x_test, features, kind='individual')
fig, ax = plt.subplots(figsize=(20, 10))
pdp_display.plot(ax=ax)
plt.show()
