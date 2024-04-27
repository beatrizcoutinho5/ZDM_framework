import numpy as np

from joblib import load


# Load model
model = load(r'models\binary\binary_random_forest_model.pkl')

# model = CatBoostClassifier()
# model.load_model(r'models\binary\binary_catboost_model.cbm')

def sample_defect_prediction_model(sample):

    # Convert into a np array
    sample_values = np.array(list(sample.values())).reshape(1, -1)

    # Defect prediction model
    prediction = model.predict_proba(sample_values)
    prediction = prediction[0]
    prediction = prediction[1]

    return prediction

# @app.route('/predict_defect')
def predict_defect(processed_sample):

    prediction = sample_defect_prediction_model(processed_sample)

    prediction = prediction*100

    prediction = round(prediction, 1)
    print(f'Prediction: {prediction}')

    return prediction
