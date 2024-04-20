import threading
import time

from flask import Flask, render_template, Request
import warnings
import json
import paho.mqtt.client as mqtt
import psycopg2

warnings.filterwarnings("ignore")

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

def on_message(client, userdata, message):

    payload = json.loads(message.payload.decode())
    recording_date = str(payload.get("Recording Date"))
    print(f"\nReceived message: {payload}")


    start_time = time.time()

    processed_sample = prepare_sample(payload)

    if processed_sample != -1:

        prediction = predict_defect(processed_sample)
        optimize_defect_score(processed_sample)

        if prediction == 1:
            shap_explainer(processed_sample)
            lime_explainer(processed_sample)

    elapsed_time = time.time() - start_time
    print(f"\nTotal Elapsed Time for sample: {elapsed_time}")

    if processed_sample != -1:
        db_save_sample(processed_sample, recording_date, prediction)

    db_get_defects_number(None, None)
    db_get_avg_feature_values()

    print(f"\n------------------------------------------------------------------------------------------------------------------------")
    print( "------------------------------------------------------------------------------------------------------------------------")


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

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':

    app.run(debug=False, threaded=True)
