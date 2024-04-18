import paho.mqtt.publish as publish
import time
import pandas as pd
import json

# MQTT broker settings
MQTT_BROKER_HOST = 'localhost'
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = 'sample_data'

# Function to publish data to MQTT topic
def publish_data():
    # Read the Excel file
    print("reading df")
    df = pd.read_excel(r'dataset_jan_dec_2022_line410.xlsx')
    print("read df")

    while True:
        # Generate a random sample
        random_sample = df.sample(n=1)

        # Convert the sample to a dictionary
        sample_dict = random_sample.to_dict(orient='records')[0]

        # Convert Timestamp objects to strings
        for key, value in sample_dict.items():
            if isinstance(value, pd.Timestamp):
                sample_dict[key] = str(value)

        # Publish data to MQTT topic
        publish.single(MQTT_TOPIC, json.dumps(sample_dict), hostname=MQTT_BROKER_HOST, port=MQTT_BROKER_PORT)
        print(f'Data published: {sample_dict}')

        # Wait for 3 minutes before publishing the next sample
        time.sleep(20)

publish_data()
