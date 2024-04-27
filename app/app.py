import threading
import time
import logging
import os
import uuid
from flask import render_template, send_from_directory
import matplotlib.pyplot as plt

from flask import Flask, render_template
from flask import request
import warnings
import json
import paho.mqtt.client as mqtt
import psycopg2
from flask import jsonify

warnings.filterwarnings("ignore")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__, template_folder='templates')

# MQTT broker settings
MQTT_BROKER_HOST = 'localhost'
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = 'sample_data'

from routes.prediction import predict_defect
from routes.optimization import optimize_defect_score, prepare_sample
from routes.explanation import shap_explainer, lime_explainer
from routes.db_analytics import db_save_sample, db_get_defects_number, db_get_historic_data, db_download_historic_data

# Database connection
DB_HOST = 'db.fe.up.pt'
DB_PORT = '5432'
DB_NAME = 'sie2338'
DB_SCHEMA = 'zdm_framework'
DB_USER = 'sie2338'
DB_PASSWORD = 'logan123'

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

shap_fig = None
lime_fig = None

line_status = "Not Working"

defects_number_result = 0
produced_panels_result = 0
percentage_defect = 0
defects_number_per_day_results = 0

explanation = 0

historic_data = []
csv_done = 0


def on_message(client, userdata, message):
    global prediction
    global defect_score_after_optim
    global optim_phrase
    global reduction_percentage

    global current_lpt
    global current_upt
    global current_pressure
    global current_tct
    global current_width
    global current_length

    global pressure_after_optim
    global upt_after_optim
    global lpt_after_optim
    global tct_after_optim

    global shap_fig
    global lime_fig

    global line_status
    global explanation

    prediction = "-"
    defect_score_after_optim = "-"
    optim_phrase = "Defect Probability After Optimization"
    reduction_percentage = "-"
    explanation = 0

    payload = json.loads(message.payload.decode())
    recording_date = str(payload.get("Recording Date"))
    print(f"\nReceived message: {payload}")

    start_time = time.time()

    if payload["Line Working?"] != -1:
        line_status = "Producing"
        print(line_status)

    processed_sample = prepare_sample(payload)

    if processed_sample == -1:
        current_lpt = "-"
        current_upt = "-"
        current_pressure = "-"
        current_tct = "-"
        current_width = "-"
        current_length = "-"
        defect_score_after_optim = reduction_percentage = tct_after_optim = pressure_after_optim = lpt_after_optim = upt_after_optim = "-"

    if processed_sample != -1:

        current_lpt = processed_sample.get('Lower Plate Temperature')
        current_upt = processed_sample.get('Upper Plate Temperature')
        current_pressure = processed_sample.get('Pressure')
        current_tct = processed_sample.get('Thermal Cycle Time')
        current_width = processed_sample.get('Width')
        current_length = processed_sample.get('Length')

        # print(current_lpt, current_upt, current_pressure, current_tct, current_width, current_length)

        prediction = predict_defect(processed_sample)

        if prediction <= 50:
            shap_fig = None
            lime_fig = None

        defect_score_after_optim = reduction_percentage = tct_after_optim = pressure_after_optim = lpt_after_optim = upt_after_optim = "load"
        optim_phrase, defect_score_after_optim, reduction_percentage, tct_after_optim, pressure_after_optim, lpt_after_optim, upt_after_optim = optimize_defect_score(
            processed_sample)

        if prediction >= 50:

            shap_fig = shap_explainer(processed_sample)
            lime_fig = lime_explainer(processed_sample)

            shap_fig.savefig('static/images/shap_plot')
            lime_fig.savefig('static/images/lime_plot')

            explanation = 1

            plt.close(shap_fig)
            plt.close(lime_fig)
    else:

        prediction = "-"

    elapsed_time = time.time() - start_time
    print(f"\nTotal Elapsed Time for sample: {elapsed_time}")

    # if processed_sample != -1:
    #     db_save_sample(processed_sample, recording_date, prediction)
    #
    # db_get_defects_number(None, None)
    # db_get_avg_feature_values()

    print(
        f"\n------------------------------------------------------------------------------------------------------------------------")
    print(
        "------------------------------------------------------------------------------------------------------------------------")


@app.route('/update-analytics')
def update_analytics():
    global defects_number_result
    global produced_panels_result
    global percentage_defect
    global defects_number_per_day_results

    from_date = request.args.get('fromDate')
    to_date = request.args.get('toDate')

    print(from_date)
    print(to_date)

    defects_number_result, produced_panels_result, percentage_defect, defects_number_per_day_results = db_get_defects_number(from_date, to_date)
    defects_number_result = defects_number_result[0]
    produced_panels_result = produced_panels_result[0]

    return jsonify({'success': True})


@app.route('/update-historic-data')
def update_historic_data():

    global historic_data
    global csv_done

    csv_done = 0

    from_date = request.args.get('fromDate')
    to_date = request.args.get('toDate')

    historic_data = db_get_historic_data(from_date, to_date)
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


@app.route('/open_dashboard_explanation')
def open_dashboard_explanation():
    return render_template('dashboard_explanation.html', shap_fig=shap_fig, lime_fig=lime_fig)


@app.route('/open_dashboard_optimization')
def open_dashboard_optimization():
    return render_template('dashboard_optimization.html', prediction=prediction, optim_phrase=optim_phrase,
                           defect_score_after_optim=defect_score_after_optim, reduce_percentage=reduction_percentage,
                           current_lpt=current_lpt, current_upt=current_upt, current_pressure=current_pressure,
                           current_width=current_width, current_length=current_length, current_tct=current_tct,
                           lpt_after_optim=lpt_after_optim, upt_after_optim=upt_after_optim,
                           tct_after_optim=tct_after_optim,
                           pressure_after_optim=pressure_after_optim,
                           explanation = explanation)

@app.route('/open_analytics')
def open_analytics():
    return render_template('analytics.html', defects_number_result=defects_number_result,
                           produced_panels_result=produced_panels_result, percentage_defect=percentage_defect,
                           defects_number_per_day_results= defects_number_per_day_results)


@app.route('/open_historic_data')
def open_historic_data():
    return render_template('historic_data.html', historic_data = historic_data, csv_done = csv_done)


@app.route('/')
def home():
    return render_template('dashboard_optimization.html', prediction=prediction, optim_phrase=optim_phrase,
                           defect_score_after_optim=defect_score_after_optim, reduce_percentage=reduction_percentage,
                           current_lpt=current_lpt, current_upt=current_upt, current_pressure=current_pressure,
                           current_width=current_width, current_length=current_length, current_tct=current_tct,
                           lpt_after_optim=lpt_after_optim, upt_after_optim=upt_after_optim,
                           tct_after_optim=tct_after_optim,
                           pressure_after_optim=pressure_after_optim)


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
