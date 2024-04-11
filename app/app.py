from flask import Flask, request, render_template, url_for, request, jsonify, redirect
import numpy as np
import pandas as pd
import warnings

from xgboost import XGBClassifier
from joblib import dump, load
from catboost import CatBoostClassifier


warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_sample', methods=['POST'])
def process_sample():

    # receive the sample data from the HTTP request
    sample_data = request.json

    # check if sample data is provided
    if not sample_data:
        return jsonify({'error': 'No sample data provided'}), 400

    # convert the sample to a NumPy array and reshape
    try:
        x = np.array(sample_data)
        x = x.reshape(1, -1)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    #
    reshaped_data = x.tolist()

    processed_sample = sample_pre_processing(reshaped_data)
    prediction = sample_defect_prediction_model(processed_sample)

    print(prediction)

    return redirect(url_for('result', prediction=prediction))

@app.route('/result.html')
def result():
    prediction = request.args.get('prediction')  # Get the prediction result from query parameters
    return render_template('result.html', prediction=prediction)


def sample_pre_processing(sample):

    # print(sample)
    sample = sample[0][0]
    # print(sample)
    # print(type(sample))


    if "Line Working?" in sample and sample["Line Working?"] == -1:
        return "Line is not working, sample not valid."

    # check for missing values in the sample
    has_missing_values = any(value is None for value in sample.values())
    if has_missing_values:
        return "The sample contains features with missing values."

    # Remove features that are not relevant for the model
    features_to_remove = ["Defect Group", "Defect Group Description", "Defect Description", "Pallet Code",
                          "Pallet Code Production Date", "Line Working?", "Humidity", "Temperature",
                          "Calculated Thermal Cycle Time", "Single Station Thermal Cycle Time",
                          "Double Station Thermal Cycle Time", "Jaw Clamping Time", "Suction Cup Pickup Time",
                          "Scratch Test", "Recording Date"]

    for feature in features_to_remove:
        if feature in sample:
            del sample[feature]

    # Transform categorical features into numerical representation
    columns_cat = ["GFFTT", "Finishing Top", "Finishing Down", "Reference Top", "Reference Down"]

    category_numerical_df = pd.read_excel('categorical_to_numeric_representations.xlsx')
    category_to_numerical_loaded = dict(
        zip(category_numerical_df['Category'], category_numerical_df['Numerical Value']))

    for feature in columns_cat:

        # print("Feature:", feature)
        # print("Original sample[feature] value:", sample[feature])

        # Check if the feature value exists in the mapping dictionary
        if sample[feature] in category_to_numerical_loaded:
            numerical_value = category_to_numerical_loaded[sample[feature]]
            sample[feature + '_cat'] = numerical_value
            # print("Numerical value:", numerical_value)

        del sample[feature]

    # Remove highly correlated features as they are not relevant for the model
    columns_remove_correlated = ["Thickness.1", "Lower Valve Temperature", "Upper Valve Temperature",
                                 "Liston 1 Speed.1", "Liston 2 Speed.1", "Floor 1.1", "Floor 2.1",
                                 "Bridge Platform.1", "Floor 1 Blow Time.1", "Floor 2 Blow Time.1",
                                 "Centering Table.1", "Finishing Down_cat", "Reference Down_cat"]

    for feature in columns_remove_correlated:
        if feature in sample:
            del sample[feature]

    # Width is assigned as an 'Object' type, so it needs to be typecasted into a numeric representation
    sample['Width'] = int(float(sample['Width']))

    return sample


def sample_defect_prediction_model(sample):

    # Extract values from the dictionary and convert them into a NumPy array
    sample_values = np.array(list(sample.values())).reshape(1, -1)

    # Now you have a NumPy array with just the values, you can use it for prediction
    # print("Sample values array:", sample_values)

    model = load(r'models\binary\binary_random_forest_model.pkl')
    prediction = model.predict(sample_values)

    return prediction

if __name__ == '__main__':
    app.run(debug=True)

