# Decision-Support Framework for Zero Defects Manufacturing

This repository implements a decision-support framework tailored to reduce defects in modern manufacturing processes. The framework addresses the escalating challenges posed by defects in industrial settings, aiming to empower manufacturing operators to achieve zero defects. Machine Learning (ML), Explainable Artificial Intelligence (XAI) and optimisation algorithms were used.

The framework was implemented in Python, with a Flask web application. HTML and JavaScript were used for the graphical components of the interface.

## Key Features

- **Real-time Defect Prediction**: Uses Machine Learning (ML) algorithms to predict defects in real-time, enabling proactive intervention. The implemented algorithms include CatBoost, Support Vector Machine (SVM), XGBoost (XGB), Random Forest (RF), and an Ensemble of RF, XGB, and CatBoost, using the respective Python packages (`scikit-learn`, `XGBoost`, `CatBoost`).

- **Explainability**: Incorporates Explainable Artificial Intelligence (XAI) methods to provide insights into the factors influencing defect occurrence. SHAP and LIME methods are implemented using the `shap` and `lime` Python packages, respectively.

- **Process Parameter Optimization**: Offers real-time suggestions for adjusting process parameters to minimize defect probability, utilizing optimization algorithms. The algorithms, including Dual Annealing, Powell method, Nelder-Mead, and Basin Hopping, are implemented using the `scipy.optimize` module.

-  **Interface for Results and Statistics**: Provides a interface to show the real-time predictions, explanations and optimization, view statistics, and analyze historical data. The interface was developed using HTML and JavaScript.

## Expected Outcomes

The decision-support framework aims to bring improvements in modern manufacturing processes, leading to benefits such as:

- **Improved Operational Efficiency:** By enabling real-time defect prediction and process parameter optimization, the framework makes operations more efficient, reducing downtime and enhancing overall productivity.

- **Reduced Material Waste:** Proactive defect prevention helps to minimise material waste, contributing to cost savings and environmental sustainability.

- **Enhanced Product Quality:** Through informed decision-making and proactive measures, the framework ensures better product quality.

- **Informed Decision-Making**: Equips manufacturing operators with insights to make informed decisions and take proactive measures to prevent defects.
  
- **Transparency and Trust**: By providing explanations for the predictions, the framework promotes transparency and trust in AI-driven decision-making processes.

## Framework Architecture

The framework structure is depicted in the figure below:

![framework_scheme](https://github.com/beatrizcoutinho5/ZDM_framework/assets/61502014/10241287-0718-4f96-b149-d41daaa8dbcf)

The prediction, explanation, and optimization features were developed as independent modules. To enhance the decision-making process, a Flask web application was developed to integrate the modules. Graphical components of the interface were developed using HTML and JavaScript. The MQTT protocol was implemented to ensure communication between the production line and the application.

Upon processing in the prediction module, cleaned samples and corresponding defect prediction are stored in a PostgreSQL database. The statistics displayed on the GUI are retrieved through SQL queries.

## **Repository Organization**

- **/dev**: Contains files and resources used for development and testing of the decision-support framework.
  - **/data**: Used data files.
  - **/plots**: Directory for storing plots generated during data analysis and model evaluation.
  - **/data_processing**: Scripts for data cleaning and preprocessing.
  - **/defect_analysis**: Scripts for analysing the patterns of defect data.
  - **/prediction**: Scripts and models related to real-time defect prediction using ML.
  - **/optimization**: Scripts for optimising process parameters, using optimisation algorithms.
  - **/xai**: Scripts with implementations for XAI methods.

- **/app**: Contains the Flask application for deploying the decision-support framework.
  - **/templates**: HTML templates for rendering the web pages.
  - **/static**: Static files such as CSS stylesheets, JavaScript, images used by the app, and other documents.
  - **/models**: To stored the trained ML models.
  - **/routes**: Python modules defining the logic for different endpoints of the app.
  - **app.py**: Main Flask application script containing route definitions and logic.
    
 ## **Dependencies**

 The required Python packages and their versions as listed below:
 
  - **catboost**: 1.2.3
  - **flask**: 3.0.2                   
  - **flask-sqlalchemy**: 3.1.1
  - **joblib**: 1.3.2
  - **lime**: 0.2.0.1
  - **matplotlib**: 3.7.0
  - **numpy**: 1.23.5
  - **paho-mqtt**: 2.0.0
  - **pandas**: 1.5.3
  - **psycopg2**: 2.9.9
  - **scikit-learn**: 1.4.1.post1
  - **scipy**: 1.10.0
  - **seaborn**: 0.12.2
  - **shap**: 0.45.0
  - **xgboost**: 2.0.3

## **Usage**

It's important to note that the application's functionality is limited without the necessary trained models and data.

### **Data Availability**

Please note that the data used for training and testing the decision-support framework is not available for public distribution and cannot be posted in this repository. If you have your own data, you can place it in the `dev/data` folder following the `.xlsx` format, and update the pre-processing, model training, and optimisation steps accordingly.

### **Running the Application**

To run the application, follow these steps:

  1. Clone or download the repository.
  2. Install the required Python packages listed above.
  3. Ensure that the trained ML models are placed in the appropriate directory (`app/models`). These models are not available in the repository.
  4. Run the Flask application by executing: `python app.py`
  5. Once the app is running, you can access the user interface by opening a web browser and navigating to [http://localhost:127.0.0.1](http://localhost:127.0.0.1).

## **Simulating MQTT Publisher**

To simulate the MQTT publisher using Mosquitto, follow these steps:

1. Ensure that Mosquitto is installed (https://mosquitto.org/download/). 
2. Run the Mosquitto broker by executing the `mosquitto.exe` file. This will allow communication between the publisher and subscriber.
3. Place the training data in `.xlsx` format in the `app/routes` directory.
4. Run the `publisher.py` file simultaneously with `app.py`.

The publisher script will read the training data from publish it using MQTT, every 70 seconds. 















