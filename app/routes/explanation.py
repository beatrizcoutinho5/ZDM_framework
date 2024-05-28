import os
import shap
import dill
import numpy as np
import pandas as pd
import lime.lime_tabular
import matplotlib.pyplot as plt

from joblib import load
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model(r'models\binary\binary_catboost_model.cbm')

# SHAP Explainer
def shap_explainer(sample):

    explainer = shap.TreeExplainer(model)

    # Get the feature names
    sample_keys = list(sample.keys())

    # Get the feature values
    sample_values = list(sample.values())
    sample_array = np.array([sample_values])

    # Perform the explainer on the sample
    sample = pd.DataFrame(sample_array, columns=sample_keys)
    shap_values_list = explainer.shap_values(sample)

    # Format shap values for plotting
    shap_values_for_class = shap_values_list[0]
    shap_values_for_class = shap_values_for_class.reshape(1, -1)
    shap_values_for_class = np.array([shap_values_for_class])
    shap_values_for_class = shap_values_for_class[:, 0, :]

    # Format sample values for plotting
    sample_array = sample_array[0]
    sample_array = sample_array.reshape(-1, 1)
    sample_array = sample_array.transpose()

    # SHAP summary plot that shows the top features importance for the prediction result
    fig, ax = plt.subplots(figsize=(25, 20))
    shap.summary_plot(shap_values_for_class, features=sample_array, feature_names=sample_keys, plot_type='bar',show=False)

    return fig


# LIME Explainer
def lime_explainer(sample):

    explainer_path = r'C:\Users\beatr\OneDrive\Ambiente de Trabalho\FACULDADE\MESTRADO\2º ANO\TESE\Código\zdm_framework\app\static\lime_explainer.pkl'

    with open(explainer_path, "rb") as f:
        explainer = dill.load(f)

    sample_values = list(sample.values())
    sample_values = np.array(sample_values)

    # Apply LIME explainer to the sample
    exp = explainer.explain_instance(
        sample_values,
        model.predict_proba
    )

    # LIME plot that displays the rules guiding the prediction
    fig = exp.as_pyplot_figure(label=1)
    fig.set_size_inches(20, 10)

    return fig


