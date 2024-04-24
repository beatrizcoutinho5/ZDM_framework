import paho.mqtt.publish as publish
import time
import pandas as pd
import json

# MQTT broker settings
MQTT_BROKER_HOST = 'localhost'
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = 'sample_data'

# publish data to MQTT topic
def publish_data():

    print("reading df")
    df = pd.read_excel(r'dataset_jan_dec_2022_line410.xlsx')
    print("read df")

    while True:

        # choose a random sample from the df
        random_sample = df.sample(n=1)

        # convert data to dict (arrives in a list)
        sample_dict = random_sample.to_dict(orient='records')[0]

        for key, value in sample_dict.items():
            if isinstance(value, pd.Timestamp):
                sample_dict[key] = str(value)

        publish.single(MQTT_TOPIC, json.dumps(sample_dict), hostname=MQTT_BROKER_HOST, port=MQTT_BROKER_PORT)
        print(f'Data published: {sample_dict}')

        # publish samples with 1 min interval
        time.sleep(40)

publish_data()
