import base64
import pandas as pd
from google.cloud.storage import Client
import datetime

def upload_to_storage(dataframe):
    storage_client = Client()
    filename = "twitter_stream"+str(datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))+".csv"
    bucket = storage_client.get_bucket("egen_twitter_project")   
    bucket.blob(filename).upload_from_string(data=dataframe.to_csv(),content_type="text/csv") 

def twitter_raw(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = pd.read_json(base64.b64decode(event['data']).decode('utf-8'))
    print(pubsub_message.iloc[0])
    upload_to_storage(pubsub_message)
    # dataframe_clean = dataframe_clean.reset_index()
    # dataframe_clean = dataframe_clean.drop(['index'], axis=1)
    # print(dataframe_clean)
    # lda_mod = topic_modelling(dataframe_clean)
    # lda_mod.train()