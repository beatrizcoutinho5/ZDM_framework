import os
import shap
import dill
import numpy as np
import pandas as pd
import lime.lime_tabular
import matplotlib.pyplot as plt

from joblib import load

# Load Model and train data
model = load(r'models\binary\binary_random_forest_model.pkl')
x_train = pd.read_excel(r'routes\binary_x_train_aug.xlsx')

# SHAP Explainer
def shap_explainer(sample):

    rf_explainer = shap.TreeExplainer(model)

    # Get the feature names
    sample_keys = list(sample.keys())

    sample = list(sample.values())
    sample = np.array(sample)

    rf_shap_values = rf_explainer.shap_values(sample)

    # Get SHAP values for the positive class (defect)
    shap_values_for_class = rf_shap_values[:, 1]

    shap_values_for_class = shap_values_for_class.reshape(-1, 1)
    shap_values_for_class = shap_values_for_class.transpose()

    sample = sample.reshape(-1, len(sample_keys))

    # SHAP summary plot that shows the top features importance for the prediction result
    fig, ax = plt.subplots(figsize=(25, 20))
    shap.summary_plot(shap_values_for_class, features=sample, feature_names=sample_keys, plot_type='bar',show=False)

    print("shap done")

    return fig


# Initialize LIME explainer with train data
# explainer = lime.lime_tabular.LimeTabularExplainer(
#         x_train.values,
#         feature_names=x_train.columns,
#         class_names=['0', '1'],
#         mode='classification',
#         discretize_continuous=True
#     )



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

    print("lime done")

    return fig


