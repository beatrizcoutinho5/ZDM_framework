import os
import numpy as np
import pandas as pd

from flask import jsonify

# Names of the features that must be present after the sample is processed
column_names = [
    'Production Line', 'Production Order Code', 'Production Order Opening', 'Length', 'Width',
    'Thickness', 'Lot Size', 'Cycle Time', 'Mechanical Cycle Time', 'Thermal Cycle Time',
    'Control Panel with Micro Stop',
    'Control Panel Delay Time', 'Sandwich Preparation Time', 'Carriage Time', 'Lower Plate Temperature',
    'Upper Plate Temperature', 'Pressure', 'Roller Start Time', 'Liston 1 Speed', 'Liston 2 Speed', 'Floor 1',
    'Floor 2', 'Bridge Platform', 'Floor 1 Blow Time', 'Floor 2 Blow Time', 'Centering Table',
    'Conveyor Belt Speed Station 1', 'Quality Inspection Cycle', 'Conveyor Belt Speed Station 2',
    'Transverse Saw Cycle',
    'Right Jaw Discharge', 'Left Jaw Discharge', 'Simultaneous Jaw Discharge', 'Carriage Speed', 'Take-off Path',
    'Stacking Cycle', 'Lowering Time', 'Take-off Time', 'High Pressure Input Time', 'Press Input Table Speed',
    'Scraping Cycle', 'Paper RC', 'Paper VC', 'Paper Shelf Life', 'GFFTT_cat', 'Finishing Top_cat',
    'Reference Top_cat'
]


def process_sample(sample_data):
    if not sample_data:
        return jsonify({'error': 'No sample data provided'}), 400

    # Convert the sample to a NumPy array and reshape (the sample arrives as a dict)
    try:
        x = np.array(sample_data)
        x = x.reshape(1, -1)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    reshaped_data = x.tolist()

    return reshaped_data


# Sample pre-processing before going into the prediction model
def sample_pre_processing(sample):
    # list
    sample = sample[0][0]

    if "Line Working?" in sample and sample["Line Working?"] == -1:
        return -1

    # Check for missing values in the sample
    if any(pd.isna(value) for value in sample.values()):
        return -1

    # Remove features that are not relevant for the model
    features_to_remove = ["Defect Group", "Defect Group Description", "Defect Description", "Pallet Code",
                          "Pallet Code Production Date", "Line Working?", "Humidity", "Temperature",
                          "Calculated Thermal Cycle Time", "Single Station Thermal Cycle Time",
                          "Double Station Thermal Cycle Time", "Jaw Clamping Time", "Suction Cup Pickup Time",
                          "Scratch Test", "Quantity", "Defect Code", "Recording Date"]

    for feature in features_to_remove:
        if feature in sample:
            del sample[feature]

    # Transform categorical features into numerical representation
    columns_cat = ["GFFTT", "Finishing Top", "Finishing Down", "Reference Top", "Reference Down"]

    # To be consistent with the model training data, the cat to num representations are stored in a file
    static_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
    file_path = os.path.join(static_folder, 'categorical_to_numeric_representations.xlsx')

    category_numerical_df = pd.read_excel(file_path)
    category_to_numerical_loaded = dict(
        zip(category_numerical_df['Category'], category_numerical_df['Numerical Value']))

    for feature in columns_cat:

        # If a categorical value in the sample doesn't already have a numeric representation defined in the file
        # the new representation is assigned (a random number between 1000 and 9999) and stored
        if sample[feature] not in category_to_numerical_loaded:

            random_value = np.random.randint(1000, 9999)

            while random_value in category_to_numerical_loaded.values():

                random_value = np.random.randint(1000, 9999)
                category_to_numerical_loaded[sample[feature]] = random_value

    # Update and save the file with the new representations, if they exist
    updated_category_numerical_df = pd.DataFrame(list(category_to_numerical_loaded.items()),columns=['Category', 'Numerical Value'])
    updated_category_numerical_df = updated_category_numerical_df.dropna(subset=['Category'])
    updated_category_numerical_df['Category'] = updated_category_numerical_df['Category'].astype(str)
    updated_category_numerical_df = updated_category_numerical_df[~updated_category_numerical_df['Category'].str.replace('.', '', regex=True).str.isdigit()]

    updated_category_numerical_df.to_excel(file_path, index=False)

    # Convert the categorical features with their numeric representation with the new updated file
    category_numerical_df = pd.read_excel(file_path)
    category_to_numerical_loaded = dict(zip(category_numerical_df['Category'], category_numerical_df['Numerical Value']))

    for feature in columns_cat:

        if sample[feature] in category_to_numerical_loaded:
            numerical_value = category_to_numerical_loaded[sample[feature]]
            sample[feature + '_cat'] = numerical_value

        # Delete the original categorical feature
        del sample[feature]

    # Remove highly correlated features as they are not relevant for the model
    columns_remove_correlated = ["Thickness.1", "Lower Valve Temperature", "Upper Valve Temperature",
                                 "Liston 1 Speed.1",
                                 "Liston 2 Speed.1", "Floor 1.1", "Floor 2.1", "Bridge Platform.1",
                                 "Floor 1 Blow Time.1",
                                 "Floor 2 Blow Time.1", "Centering Table.1", "Finishing Down_cat", "Reference Down_cat"]

    for feature in columns_remove_correlated:
        if feature in sample:
            del sample[feature]

    # Width is assigned as an 'Object' type, so it needs to be typecasted into a numeric representation
    sample['Width'] = int(float(sample['Width']))

    # Checks if the sample has all the features that are needed for the model prediction
    missing_columns = [col for col in column_names if col not in sample.keys()]

    if missing_columns:

        print("Sample is missing the following features:")
        for col in missing_columns:
            print(col)

        return -1

    # Pre-processed sample
    return sample


def prepare_sample(sample_data):

    sample = process_sample(sample_data)
    processed_sample = sample_pre_processing(sample)

    if processed_sample == -1:
        return -1
    else:
        return processed_sample
