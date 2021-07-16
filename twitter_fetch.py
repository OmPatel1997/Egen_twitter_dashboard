import requests
import os
import json
import pandas as pd
import re

import logging
from google.cloud.storage import Client
from concurrent import futures
from google.cloud.pubsub_v1 import PublisherClient
from google.cloud.pubsub_v1.publisher.futures import Future
# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
# bearer_token = os.environ.get("BEARER_TOKEN")

class fetch_data(object):
    def __init__(self):
        self.bearer_token = 'AAAAAAAAAAAAAAAAAAAAAI3gRgEAAAAAUPLlRPGTe%2FDMYomOO2AjFLFcDKM%3DhozMlnRXNYdmVzc9B98aUijRDY3M4A59D5Q7p87izhvVKV8vg0'
        self.url = "https://api.twitter.com/2/tweets/sample/stream"

    def bearer_oauth(self,r):
        """
        Method required by bearer token authentication.
        """

        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2SampledStreamPython"
        return r

    def connect_to_endpoint(self):
        response = requests.request("GET", self.url, auth=self.bearer_oauth, stream=True)
        print(response.status_code)
        count = 0
        # print(len(response.iter_lines))
        dframe = pd.DataFrame()
        for response_line in response.iter_lines():
            if response_line:
                count+=1
                json_response = json.loads(response_line)
                dframe = dframe.append(pd.json_normalize(json_response['data']))
                if count%10 ==0:
                    print("number of tweets = ", str(count))
                    break
                # print(len(json.dumps(json_response, indent=4, sort_keys=True)))
        print("number of tweets = ", str(count))
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )

        return dframe

class api_to_pubsub:
    def __init__(self):
        self.project_id = "logical-flame-319700"
        self.topic_id = "twitter_pubsub"
        self.publisher = PublisherClient()
        self.topic_path = self.publisher.topic_path(self.project_id, self.topic_id)
        self.publish_futures = []
        pass


    def get_callback(self, publish_future, data):
        def callback(publish_future):
            try:
                # Wait 60 seconds for the publish call to succeed.
                print(publish_future.result(timeout=60))
            except futures.TimeoutError:
                print("Publishing {data} timed out.")

        return callback
        
    def send_to_topic(self, message):
        publish_future = self.publisher.publish(self.topic_path, message.encode("utf-8"))
        # Non-blocking. Publish failures are handled in the callback function.
        publish_future.add_done_callback(self.get_callback(publish_future, message))
        self.publish_futures.append(publish_future)

        # Wait for all the publish futures to resolve before exiting.
        futures.wait(self.publish_futures, return_when=futures.ALL_COMPLETED)

def main():
    fetch = fetch_data()
    dataframe_raw = fetch.connect_to_endpoint()
    # print(dataframe_raw)
    # clean = clean_data(dataframe_raw)
    # dataframe_clean = clean.process()
    # print(dataframe_clean)
    dataframe_raw = dataframe_raw.reset_index()
    dataframe_raw = dataframe_raw.drop(['index'], axis=1)
    # print(dataframe_clean)
    # lda_mod = topic_modelling(dataframe_clean)
    # lda_mod.train()
    print(dataframe_raw)
    publish_object = api_to_pubsub()
    publish_object.send_to_topic(dataframe_raw.to_json())


if __name__ == "__main__":
    main()