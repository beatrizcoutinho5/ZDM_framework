# Decision-Support Framework for Zero Defects Manufacturing

This repository contains the implementation of a decision-support framework tailored to reduce defects in modern manufacturing processes. The project was developed as part of a Master's dissertation in Electrical and Computer Engineering at FEUP.

## Table of Contents
1. [Framework Overview and Architecture](#framework-overview-and-architecture)
2. [Key Features and Implementation](#key-features-and-implementation)
3. [Expected Outcomes](#expected-outcomes)
4. [ML Models and Optimization Results](#ml-models-and-optimization-results)
5. [Repository Organization](#repository-organization)
6. [Dependencies](#dependencies)
7. [Usage](#usage)
    - [Data Availability](#data-availability)
    - [Running the Application](#running-the-application)
    - [Simulating MQTT Publisher](#simulating-mqtt-publisher)
    - [UI Demonstration](#ui-demonstration)

  
## Framework Overview and Architecture

The developed framework allows defect prediction, enabling the proactive anticipation of defective products, by applying data-driven Machine Learning (ML) algorithms. Additionally, it incorporates recommendations for adjusting process parameters to mitigate defects and enhance overall process efficiency, using optimization algorithms. Lastly, the framework encompasses defect explanation, clarifying the underlying factors of defects resorting to Explainable Artificial Intelligence (XAI) methods. Besides the preventive and predictive models, the tool incorporates a PostgreSQL database for efficient data storage and graphical user interface (GUI) for the visualization of results. The framework was implemented in Python, with a Flask web application. HTML and JavaScript were used for the graphical components of the interface.

The framework structure is depicted in the figure below:

<img src="https://github.com/beatrizcoutinho5/ZDM_framework/assets/61502014/10241287-0718-4f96-b149-d41daaa8dbcf" alt="framework_scheme" width="400">

The prediction, explanation, and optimization features were developed as seperate modules. To enhance the decision-making process, a Flask web application was developed to integrate the the three modules. The MQTT protocol was implemented to ensure communication between the manufacturing site and the application. The UML sequence diagram illustrates the interactions between the various framework modules within the web application:

![uml_sequence_diagram_new](https://github.com/beatrizcoutinho5/ZDM_framework/assets/61502014/30acf1d2-d2d2-4d28-a7e1-ccf6a0902f0e)


## Key Features and Implementation

The key features of this framework and their respective implementations are briefly describide bellow:

- **Real-time Defect Prediction**: Uses Machine Learning (ML) algorithms to predict defects in real-time, enabling proactive intervention. The implemented algorithms include CatBoost, Support Vector Machine (SVM), XGBoost (XGB), Random Forest (RF), and an Ensemble of RF, XGB, and CatBoost, using the respective Python packages (`scikit-learn`, `XGBoost`, `CatBoost`).

- **Explainability**: Incorporates XAI methods to provide insights into the factors influencing defect occurrence. SHAP and LIME methods are implemented using the `shap` and `lime` Python packages, respectively.

- **Process Parameter Optimization**: Offers real-time suggestions for adjusting process parameters to minimize defect probability, utilizing optimization algorithms including Dual Annealing, Powell method, Nelder-Mead, and Basin Hopping, implemented using the `scipy.optimize` module.

-  **Interface for Results and Statistics**: Provides a interface to show users the real-time predictions, explanations and optimization, view statistics, and analyze historical data. The interface was developed using HTML and JavaScript.

## Expected Outcomes

The decision-support framework aims to bring improvements in modern manufacturing processes, leading to benefits such as:

- **Improved Operational Efficiency:** By enabling real-time defect prediction and process parameter optimization, the framework makes operations more efficient, reducing downtime and enhancing overall productivity.

- **Reduced Material Waste:** Proactive defect prevention helps to minimise material waste, contributing to cost savings and environmental sustainability.

- **Enhanced Product Quality:** Through informed decision-making and proactive measures, the framework ensures better product quality.

- **Informed Decision-Making**: Equips manufacturing operators with insights to make informed decisions and take proactive measures to prevent defects.
  
- **Transparency and Trust**: By providing explanations for the predictions, the framework promotes transparency and trust in AI-driven decision-making processes.

## ML Models and Optimization Results

### Prediction Using ML Models

Supervised learning methods were implemented, focusing on classification tasks. The SMOTE algorithm was applied to the training data to deal with imbalanced data. The resuls are presented in the table bellow:
![prediction_results](https://github.com/beatrizcoutinho5/ZDM_framework/assets/61502014/875850c8-9901-4587-b918-94d512b67b9c)




### Optimization

Four optimisation algorithms were tested to find optimal process parameters: Dual Annealing, Nelder-Mead, Powell, and Basin Hopping. Each algorithm requires a fitness function to minimise the defect probability. Three types of functions were compared: Mean Squared Error (MSE), LogCosh, and Mean Absolute Error (MAE), each evaluated with different target defect probability values (0%, 10%, and 50%). Additionally, MSE was evaluated without a specific target defect probability. The resuls are presented in the table bellow:
![optimisation_results](https://github.com/beatrizcoutinho5/ZDM_framework/assets/61502014/07931979-290b-45f0-8a31-1ecc24d2c6b3)


## **Repository Organization**

The repository is organized as follows:

- **`/dev`**: Contains files and resources used for development and testing of the decision-support framework.
  - **`/data`**: Used data files.
  - **`/plots`**: Directory for storing plots generated during data analysis and model evaluation.
  - **`/data_processing`**: Scripts for data cleaning and preprocessing.
  - **`/defect_analysis`**: Scripts for analysing the patterns of defect data.
  - **`/prediction`**: Scripts and models related to real-time defect prediction using ML.
  - **`/optimization`**: Scripts for optimising process parameters, using optimisation algorithms.
  - **`/xai`**: Scripts with implementations for XAI methods.

- **`/app`**: Contains the Flask application for deploying the decision-support framework.
  - **`/templates`**: HTML templates for rendering the web pages.
  - **`/static`**: Static files such as CSS stylesheets, JavaScript, images, and other documents used by the app.
  - **`/models`**: To stored the trained ML models.
  - **`/routes`**: Python modules defining the logic for different endpoints of the app.
  - **`app.py`**: Main Flask application script containing route definitions and logic.
    
 ## **Dependencies**

 Python 3.10.9 was used to develop tge project. The required packages and their versions are listed below:
 
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

The data used for training and testing the decision-support framework is not available for public distribution and cannot be posted in this repository. If you have your own data, you can place it in the `dev/data` folder following the `.xlsx` format, and update the pre-processing, model training, and optimisation steps accordingly.

### **Running the Application**

To run the application, follow these steps:

  1. Clone or download the repository.
  2. Install the required Python packages listed above.
  3. Ensure that the trained ML models are placed in the appropriate directory (`app/models`).
  4. Run the Flask application by executing: `python app.py`
  5. Once the app is running, you can access the UI by opening a web browser and navigating to [http://localhost:5000](http://localhost:5000). Please note that optimal UI performance is achivied using Google Chrome with screen resolution of 1920px x 1080 and 125% scale.

### **Simulating MQTT Publisher**

To simulate the MQTT publisher using Mosquitto, follow these steps:

1. Ensure that [Mosquitto](https://mosquitto.org/download/) is installed. 
2. Run the Mosquitto broker by executing the `mosquitto.exe` file. This will allow communication between the publisher and subscriber.
3. Place the training data in `.xlsx` format in the `app/routes` directory.
4. Run the `publisher.py` file (placed in `app/routes`) simultaneously with `app.py`.

The publisher script will read the training data from publish it using MQTT, every 70 seconds.

### **UI Demonstration**

Below is a demonstration of the user interface, showcasing the real-time prediction, explanation, and optimization features integrated into the decision-support framework.

















