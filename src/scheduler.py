import os
import glob
import requests
import time
import logging
import psycopg2
from configparser import ConfigParser

from io_utils import json_io

import image_set_model






def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def update_status(cur, conn, key, status):

    sql = """ UPDATE detection_models
              SET status = %s 
              WHERE model_id = %s"""

    cur.execute(sql, (key, status))
    # records = [row for row in cur.fetchall()]
    conn.commit()
    cur.close()

def loop():

    logger = logging.getLogger(__name__)
    logger.info("Scheduler started")

    # params = config()
    # conn = psycopg2.connect(**params)

    # #conn = psycopg2.connect("dbname=plant_detection_db user=plant_detection_dbuser")
    
    # logger.info("Connected to plant_detection_db")

    # cur = conn.cursor()


    while True:


        prediction_requests_dir = os.path.join("usr", "data", "prediction_requests")

        prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))

        while len(prediction_request_paths) > 0:
            print("there is a prediction to process", prediction_request_paths)

            prediction_request_path = prediction_request_paths[0]
            print("prediction_request_path", prediction_request_path)
            prediction_request = json_io.load_json(prediction_request_path)

            farm_name = prediction_request["farm_name"]
            field_name = prediction_request["field_name"]
            mission_date = prediction_request["mission_date"]


            image_set_model.handle_prediction_request(prediction_request)



            os.remove(prediction_request_path)
            prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))


        
        for farm_path in glob.glob(os.path.join("usr", "data", "image_sets", "*")):
            farm_name = os.path.basename(farm_path)
            for field_path in glob.glob(os.path.join(farm_path, "*")):
                field_name = os.path.basename(field_path)
                for mission_path in glob.glob(os.path.join(field_path, "*")):
                    mission_date = os.path.basename(mission_path)

                    image_set_model.check_train(farm_name, field_name, mission_date)


        time.sleep(0.1)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    loop()