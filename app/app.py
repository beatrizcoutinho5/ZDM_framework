import time
import json
import logging
import warnings
import threading
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

from flask import Flask, render_template, jsonify, request

from routes.prediction import predict_defect
from routes.dataprocessing import prepare_sample
from routes.explanation import shap_explainer, lime_explainer
from routes.optimization import optimize_defect_score
from routes.db_analytics import db_save_sample, db_get_defects_number, db_get_historic_data, db_download_historic_data

# Suppressing warnings and ignoring server logs
warnings.filterwarnings("ignore")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__, template_folder='templates')

# MQTT broker settings
MQTT_BROKER_HOST = 'localhost'
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = 'sample_data'

# variables initialization

active_page = 'dashboard'

# for optimization dashboard
prediction = "-"
defect_score_after_optim = "-"
optim_phrase = "Defect Probability After Optimization"
reduction_percentage = "-"

current_lpt = "-"
current_upt = "-"
current_pressure = "-"
current_tct = "-"
current_width = "-"
current_length = "-"

pressure_after_optim = "-"
upt_after_optim = "-"
lpt_after_optim = "-"
tct_after_optim = "-"

explanation = 0

# for explanation dashboard
shap_fig = None
lime_fig = None

# for analytics page
line_status = "Not Working"

defects_number_result = 0
produced_panels_result = 0
percentage_defect = 0
defects_number_per_day_results = 0

# for historic data page

historic_data = []
csv_done = 0


# when a mqqt message is received
def on_message(client, userdata, message):
    global prediction, defect_score_after_optim, optim_phrase, reduction_percentage
    global current_lpt, current_upt, current_pressure, current_tct, current_width, current_length
    global pressure_after_optim, upt_after_optim, lpt_after_optim, tct_after_optim
    global explanation, shap_fig, lime_fig, line_status

    # # variable initialization
    # prediction = "-"
    # defect_score_after_optim = "-"
    # optim_phrase = "Defect Probability After Optimization"
    # reduction_percentage = "-"
    # explanation = 0

    payload = json.loads(message.payload.decode())
    recording_date = str(payload.get("Recording Date"))

    # Start timer
    start_time = time.time()

    # Get production line status
    if payload["Line Working?"] != -1:
        line_status = "Producing"

    # Process the raw data received
    processed_sample = prepare_sample(payload)

    if processed_sample == -1:

        # Invalid sample (missing values), set values as '-'
        current_lpt = "-"
        current_upt = "-"
        current_pressure = "-"
        current_tct = "-"
        current_width = "-"
        current_length = "-"
        defect_score_after_optim = reduction_percentage = tct_after_optim = pressure_after_optim = lpt_after_optim = upt_after_optim = "-"

    elif processed_sample != -1:

        # Valid sample, update current feature values
        current_lpt = processed_sample.get('Lower Plate Temperature')
        current_upt = processed_sample.get('Upper Plate Temperature')
        current_pressure = processed_sample.get('Pressure')
        current_tct = processed_sample.get('Thermal Cycle Time')
        current_width = processed_sample.get('Width')
        current_length = processed_sample.get('Length')

        # Predict defect probability
        prediction = predict_defect(processed_sample)

        # Set the "load" condition to display a loading image on the UI while the optimization model is running,
        # as it can take up to 1 minute to complete
        defect_score_after_optim = reduction_percentage = tct_after_optim = pressure_after_optim = lpt_after_optim = upt_after_optim = "load"

        # Optimize defect score
        optim_phrase, defect_score_after_optim, reduction_percentage, tct_after_optim, pressure_after_optim, lpt_after_optim, upt_after_optim = optimize_defect_score(
            processed_sample)

        # Generate explanation if sample likely to be a defect
        if prediction >= 50:

            shap_fig = shap_explainer(processed_sample)
            lime_fig = lime_explainer(processed_sample)

            shap_fig.savefig('static/images/shap_plot')
            lime_fig.savefig('static/images/lime_plot')

            explanation = 1

            plt.close(shap_fig)
            plt.close(lime_fig)

        else:

            shap_fig = None
            lime_fig = None


    # Calculate and print the elapsed time for the processing, prediction, optimization, and explanation pipeline
    elapsed_time = time.time() - start_time
    print(f"\nTotal Elapsed Time for sample: {elapsed_time}")

    if processed_sample != -1:

        # Save processed sample and prediction to database
        db_save_sample(processed_sample, recording_date, prediction)

    print(f"\n--------------------------------------------------------------------------------------------------------")


# Endpoint to update the analytics page values
@app.route('/update-analytics')
def update_analytics():

    global defects_number_result, produced_panels_result, percentage_defect, defects_number_per_day_results

    # Get the start and end dates entered by the user
    from_date = request.args.get('fromDate')
    to_date = request.args.get('toDate')

    # Call the function to retrieve defects count, produced panels count,
    # defect percentage, and defects count per day for the selected date range
    defects_number_result, produced_panels_result, percentage_defect, defects_number_per_day_results = db_get_defects_number(
        from_date, to_date)

    defects_number_result = defects_number_result[0]
    produced_panels_result = produced_panels_result[0]

    return jsonify({'success': True})


# Endpoint to update the historic data page values
@app.route('/update-historic-data')
def update_historic_data():

    global historic_data, csv_done
    csv_done = 0

    # Get the start and end dates entered by the user
    from_date = request.args.get('fromDate')
    to_date = request.args.get('toDate')

    # Retrieve the 3 most recent samples for the selected time period for UI displa
    historic_data = db_get_historic_data(from_date, to_date)

    # Convert all the database samples within the selected time period with all features into
    # a CSV file ready to download
    csv_done = db_download_historic_data(from_date, to_date)

    return jsonify({'success': True})


# MQTT client
def mqtt_loop():

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
    mqtt_client.subscribe(MQTT_TOPIC)
    mqtt_client.loop_forever()


mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.start()


# Render page templates
@app.route('/open_dashboard_explanation')
def open_dashboard_explanation():

    return render_template('dashboard_explanation.html', shap_fig=shap_fig, lime_fig=lime_fig, active_page='dashboard')

@app.route('/open_dashboard_optimization')
def open_dashboard_optimization():

    return render_template('dashboard_optimization.html', prediction=prediction, optim_phrase=optim_phrase,
                           defect_score_after_optim=defect_score_after_optim, reduce_percentage=reduction_percentage,
                           current_lpt=current_lpt, current_upt=current_upt, current_pressure=current_pressure,
                           current_width=current_width, current_length=current_length, current_tct=current_tct,
                           lpt_after_optim=lpt_after_optim, upt_after_optim=upt_after_optim,
                           tct_after_optim=tct_after_optim, pressure_after_optim=pressure_after_optim,
                           explanation=explanation, active_page='dashboard')

@app.route('/open_analytics')
def open_analytics():

    return render_template('analytics.html', defects_number_result=defects_number_result,
                           produced_panels_result=produced_panels_result, percentage_defect=percentage_defect,
                           defects_number_per_day_results=defects_number_per_day_results, active_page='analytics')

@app.route('/open_historic_data')
def open_historic_data():

    return render_template('historic_data.html', historic_data=historic_data, csv_done=csv_done,
                           active_page='historic_data')


# Default page - Optimization Dashboard
@app.route('/')
def home():

    return render_template('dashboard_optimization.html', prediction=prediction, optim_phrase=optim_phrase,
                           defect_score_after_optim=defect_score_after_optim, reduce_percentage=reduction_percentage,
                           current_lpt=current_lpt, current_upt=current_upt, current_pressure=current_pressure,
                           current_width=current_width, current_length=current_length, current_tct=current_tct,
                           lpt_after_optim=lpt_after_optim, upt_after_optim=upt_after_optim,
                           tct_after_optim=tct_after_optim, pressure_after_optim=pressure_after_optim,
                           active_page='dashboard')

# Run app
if __name__ == '__main__':
    app.run(debug=False, threaded=True)
