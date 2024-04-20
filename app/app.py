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

# Database connection
DB_HOST = 'db.fe.up.pt'
DB_PORT = '5432'
DB_NAME = 'sie2338'
DB_SCHEMA = 'zdm_framework'
DB_USER = 'sie2338'
DB_PASSWORD = 'logan123'

def connect_to_database():
    try:
        conn = psycopg2.connect(
            dbname=f"{DB_NAME} {DB_SCHEMA}",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("Connected to the database")
        return conn
    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
        return None


def on_message(client, userdata, message):

    payload = json.loads(message.payload.decode())
    recording_date = str(payload.get("Recording Date"))
    print("------------------------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"\nReceived message: {payload}")


    start_time = time.time()

    processed_sample = prepare_sample(payload)

    if processed_sample != -1:
        prediction = predict_defect(processed_sample)
        optimize_defect_score(processed_sample)
        print(prediction)

        if prediction == 1:
            shap_explainer(processed_sample)
            lime_explainer(processed_sample)

    elapsed_time = time.time() - start_time
    print(f"\nTotal Elapsed Time for sample: {elapsed_time}")

    # db connect
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    if processed_sample != -1:

        # order of keys
        order = ["Production Line", "Production Order Code", "Production Order Opening", "Length", "Width", "Thickness",
                 "Lot Size",
                 "Cycle Time", "Mechanical Cycle Time", "Thermal Cycle Time", "Control Panel with Micro Stop",
                 "Control Panel Delay Time", "Sandwich Preparation Time", "Carriage Time", "Lower Plate Temperature",
                 "Upper Plate Temperature",
                 "Pressure", "Roller Start Time", "Liston 1 Speed", "Liston 2 Speed", "Floor 1", "Floor 2",
                 "Bridge Platform", "Floor 1 Blow Time",
                 "Floor 2 Blow Time", "Centering Table", "Conveyor Belt Speed Station 1", "Quality Inspection Cycle",
                 "Conveyor Belt Speed Station 2",
                 "Transverse Saw Cycle", "Right Jaw Discharge", "Left Jaw Discharge", "Simultaneous Jaw Discharge",
                 "Carriage Speed", "Take-off Path",
                 "Stacking Cycle", "Lowering Time", "Take-off Time", "High Pressure Input Time",
                 "Press Input Table Speed", "Scraping Cycle", "Paper RC",
                 "Paper VC", "Paper Shelf Life", "GFFTT_cat", "Finishing Top_cat", "Reference Top_cat"]

        # order the processed sample
        ordered_processed_sample = {key: processed_sample[key] for key in order if key in processed_sample}

        # SQL query
        columns = ', '.join(['"Recording Date"' if recording_date else '',
                             '"Defect Prediction"' if prediction else ''] +
                            [f'"{col}"' for col in ordered_processed_sample.keys()])
        placeholders = ', '.join(['%s'] * (len(ordered_processed_sample) + 2))
        query = f"INSERT INTO zdm_framework.ProductionData ({columns}) VALUES ({placeholders})"

        # values from the processed sample
        values = list(ordered_processed_sample.values())
        values.insert(0, recording_date)
        values.insert(1, prediction)

        cursor.execute(query, values)

        conn.commit()
        cursor.close()
        conn.close()
        print(f"\nSaved sample data to database!")
        print("------------------------------------------------------------------------------------------------------------------------")
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
