import os
import glob
import shutil
#import requests
import time
import logging
#import psycopg2

import tensorflow as tf

#from configparser import ConfigParser

from io_utils import json_io

import image_set_actions as isa
import image_set_model as ism

def check_train():

    for username_path in glob.glob(os.path.join("usr", "data", "*")):
        username = os.path.basename(username_path)
        for user_dir in glob.glob(os.path.join(username_path, "*")):
            if os.path.basename(user_dir) == "image_sets":
                for farm_path in glob.glob(os.path.join(username_path, "image_sets", "*")):
                    farm_name = os.path.basename(farm_path)
                    for field_path in glob.glob(os.path.join(farm_path, "*")):
                        field_name = os.path.basename(field_path)
                        for mission_path in glob.glob(os.path.join(field_path, "*")):
                            mission_date = os.path.basename(mission_path)

                            try:
                                ism.check_train(username, farm_name, field_name, mission_date)
                            except FileNotFoundError:
                                pass




def check_predict():
    for username_path in glob.glob(os.path.join("usr", "data", "*")):
        username = os.path.basename(username_path)
        for user_dir in glob.glob(os.path.join(username_path, "*")):
            if os.path.basename(user_dir) == "image_sets":
                for farm_path in glob.glob(os.path.join(username_path, "image_sets", "*")):
                    farm_name = os.path.basename(farm_path)
                    for field_path in glob.glob(os.path.join(farm_path, "*")):
                        field_name = os.path.basename(field_path)
                        for mission_path in glob.glob(os.path.join(field_path, "*")):
                            mission_date = os.path.basename(mission_path)

                            
                            ism.check_predict(username, farm_name, field_name, mission_date)



# def check_predict():

#     prediction_requests_dir = os.path.join("usr", "requests", "prediction")

#     prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))

#     while len(prediction_request_paths) > 0:
#         # print("there is a prediction to process", prediction_request_paths)

#         prediction_request_path = prediction_request_paths[0]
#         # print("prediction_request_path", prediction_request_path)
        

#         # farm_name = prediction_request["farm_name"]
#         # field_name = prediction_request["field_name"]
#         # mission_date = prediction_request["mission_date"]


#         ism.handle_prediction_request(prediction_request_path)



        
#         prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))



def check_baseline():
    for farm_path in glob.glob(os.path.join("usr", "data", "image_sets", "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                model_dir = os.path.join(mission_path, "model")
                initialize_request_path = os.path.join(model_dir, "initialize.json")
                if os.path.exists(initialize_request_path):
                    initialize_request = json_io.load_json(initialize_request_path)
                    ism.handle_baseline_request(initialize_request)
                    #os.remove(initialize_request_path)
                    shutil.move(initialize_request_path, os.path.join(model_dir, "tmp_initialize.json"))
                    #exit()

    # baseline_requests_dir = os.path.join("usr", "requests", "baseline")

    # baseline_request_paths = glob.glob(os.path.join(baseline_requests_dir, "*.json"))

    # while len(baseline_request_paths) > 0:

    #     baseline_request_path = baseline_request_paths[0]
    #     baseline_request = json_io.load_json(baseline_request_path)

    #     ism.handle_baseline_request(baseline_request)

    #     os.remove(baseline_request_path)
    #     baseline_request_paths = glob.glob(os.path.join(baseline_requests_dir, "*.json"))


def loop():

    logger = logging.getLogger(__name__)
    logger.info("Scheduler started")

    # params = config()
    # conn = psycopg2.connect(**params)

    # #conn = psycopg2.connect("dbname=plant_detection_db user=plant_detection_dbuser")
    
    # logger.info("Connected to plant_detection_db")

    # cur = conn.cursor()

    


    while True:
        # isa.check_restart()

        # check_baseline()

        check_predict()

        check_train()

        # check_baseline()

        time.sleep(0.1)


if __name__ == "__main__":

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    # if gpus:
    # # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         # tf.config.set_logical_device_configuration(
    #         #     gpus[0],
    #         #     [tf.config.LogicalDeviceConfiguration(memory_limit=(16*1024))])
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    logging.basicConfig(level=logging.INFO)
    loop()