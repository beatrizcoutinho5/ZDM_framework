import threading
import time
import logging

from flask import Flask, render_template, Request
import warnings
import json
import paho.mqtt.client as mqtt
import psycopg2

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
from routes.db_analytics import db_save_sample, db_get_defects_number, db_get_avg_feature_values

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

    prediction = "-"
    defect_score_after_optim = "-"
    optim_phrase = "Defect Probability After Optimization"
    reduction_percentage = "-"

    payload = json.loads(message.payload.decode())
    recording_date = str(payload.get("Recording Date"))
    print(f"\nReceived message: {payload}")

    start_time = time.time()

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

        print(current_lpt, current_upt, current_pressure, current_tct, current_width, current_length)

        prediction = predict_defect(processed_sample)

        defect_score_after_optim = reduction_percentage = tct_after_optim = pressure_after_optim = lpt_after_optim = upt_after_optim = "load"
        optim_phrase, defect_score_after_optim, reduction_percentage, tct_after_optim, pressure_after_optim, lpt_after_optim, upt_after_optim = optimize_defect_score(processed_sample)

        if prediction >= 50:
            shap_explainer(processed_sample)
            lime_explainer(processed_sample)
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


# MQTT client
def mqtt_loop():
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
    mqtt_client.subscribe(MQTT_TOPIC)
    mqtt_client.loop_forever()


mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.start()


# init_prediction_routes(app)
# init_optimization_routes(app)
# init_explanation_routes(app)

@app.route('/dashboard_explanation')
def dashboard_explanation():
    return render_template('dashboard_explanation.html')


@app.route('/')
def home():
    return render_template('dashboard_optimization.html', prediction=prediction, optim_phrase=optim_phrase,
                           defect_score_after_optim=defect_score_after_optim, reduce_percentage=reduction_percentage,
                           current_lpt=current_lpt, current_upt=current_upt, current_pressure=current_pressure,
                           current_width=current_width, current_length=current_length, current_tct=current_tct,
                           lpt_after_optim = lpt_after_optim, upt_after_optim = upt_after_optim, tct_after_optim= tct_after_optim,
                           pressure_after_optim = pressure_after_optim)



if __name__ == '__main__':
    app.run(debug=False, threaded=True)
