import logging
import os
import tensorflow as tf

from io_utils import json_io
import image_set_model as ism

def resume_baseline_request(baseline_name):

    logging.basicConfig(level=logging.INFO)
    ism.handle_resume_direct_baseline_request(baseline_name)


def run_baseline_request():

    logging.basicConfig(level=logging.INFO)

    request = {
        "baseline_name": "p2irc_baseline", #full_yolov4_8_sets_0",
        #                  "UNI::LowN1::2021-06-07::"
        #                  "BlaineLake::HornerWest::2021-06-09::" +
        #                  "Saskatoon::Norheim1::2021-05-26::" +
        #                  "UNI::Sutherland::2021-06-05::" +
        #                  "UNI::Brown::2021-06-05::" + 
        #                  "row_spacing::brown_low_res::2021-06-01::" +
        #                  "row_spacing::nasser_low_res::2020-06-08::" +
        #                  "Saskatoon::Norheim2::2021-05-26::" +
        #                  "Saskatoon::Norheim4::2022-05-24",
        #"baseline_name": "all_test",
        #"baseline_name": "UNI::Sutherland::2021-06-05",
        
        "image_sets": [
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "LowN1",
            "mission_date": "2021-06-07"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "River",
            "mission_date": "2021-06-09"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "HornerWest",
            "mission_date": "2021-06-09"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim1",
            "mission_date": "2021-05-26"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "Sutherland",
            "mission_date": "2021-06-05"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "Brown",
            "mission_date": "2021-06-05"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim2",
            "mission_date": "2021-05-26"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "LowN2",
            "mission_date": "2021-06-07"
        },
        {
            "username": "kaylie",
            "farm_name": "Biggar",
            "field_name": "Dennis1",
            "mission_date": "2021-06-04"
        },
        # {
        #     "username": "kaylie",
        #     "farm_name": "row_spacing",
        #     "field_name": "nasser",
        #     "mission_date": "2021-06-01"
        # },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "Lake",
            "mission_date": "2021-06-09"
        },
        # {
        #     "username": "kaylie",
        #     "farm_name": "row_spacing",
        #     "field_name": "brown",
        #     "mission_date": "2021-06-01"
        # },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim1",
            "mission_date": "2021-06-02"
        },
        {
            "username": "kaylie",
            "farm_name": "MORSE",
            "field_name": "Dugout",
            "mission_date": "2022-05-27"
        },
        {
            "username": "kaylie",
            "farm_name": "MORSE",
            "field_name": "Nasser",
            "mission_date": "2022-05-27"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "Dugout",
            "mission_date": "2022-05-30"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim4",
            "mission_date": "2022-05-24"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim5",
            "mission_date": "2022-05-24"
        }

        ]
    }

    for image_set in request["image_sets"]:
        annotations_path = os.path.join("usr", "data", image_set["username"], "image_sets",
                           image_set["farm_name"], image_set["field_name"], image_set["mission_date"],
                           "annotations", "annotations_w3c.json")
        annotations = json_io.load_json(annotations_path)
        image_set["images"] = [image_name for image_name in annotations.keys() if annotations[image_name]["status"] == "completed_for_training"]
        # image_set["images"] = [image_set["images"][0]]


    ism.handle_direct_baseline_request(request)

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
    #         # for gpu in gpus:
    #         #     tf.config.experimental.set_memory_growth(gpu, True)
    #         tf.config.set_logical_device_configuration(
    #             gpus[0],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit=(12*1024))])
    #         # logical_gpus = tf.config.list_logical_devices('GPU')
    #         # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)
    run_baseline_request()