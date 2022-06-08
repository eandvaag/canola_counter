import os
import glob
import shutil
import argparse
import imagesize
import requests

import random
import numpy as np


from io_utils import json_io, w3c_io, tf_record_io

import extract_patches as ep
from image_set import Image

from models.yolov4 import yolov4_image_set_driver


IDLE = "idle"
TRAINING = "training"
PREDICTING = "predicting"



notification_url = "http://172.16.1.71:8110/plant_detection/notification"


def notify(farm_name, field_name, mission_date, extra_items={}):
    #print("sending data", data)


    status_path = os.path.join("usr", "data", "image_sets",
                                farm_name, field_name, mission_date, 
                                "model", "status.json")
    status = json_io.load_json(status_path)

    data = {
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date
    }
    for k, v in status.items():
        data[k] = v
    for k, v in extra_items.items():
        data[k] = v

    response = requests.post(notification_url, data=data)
    response = response.json()
    if response["message"] != "received":
        print("request not received")
        print(response)
        exit()



def set_status(status, farm_name, field_name, mission_date):

    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")

    status_config_path = os.path.join(model_dir, "status.json")
    status_config = json_io.load_json(status_config_path)

    status_config["status"] = status

    json_io.save_json(status_config_path, status_config)



def create_patches_if_needed(farm_name, field_name, mission_date):
    
    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    images_dir = os.path.join(image_set_dir, "images")
    patches_dir = os.path.join(image_set_dir, "patches")
    patch_data_path = os.path.join(patches_dir, "patch_data.json")

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    try:
        patch_size = w3c_io.get_patch_size(annotations)
    except RuntimeError:
        patch_size = 300
    print("patch_size", patch_size)

    if os.path.exists(patch_data_path):
        existing_patch_size = imagesize.get(glob.glob(os.path.join(patches_dir, "*.png"))[0])[0]
        print("existing patch size", existing_patch_size)
        abs_patch_size_diff = abs(existing_patch_size - patch_size)

    update_thresh = 10

    if not os.path.exists(patch_data_path) or abs_patch_size_diff >= update_thresh:
        patch_data = {}

        print("need to extract patches")
        shutil.rmtree(patches_dir)
        os.mkdir(patches_dir)


        for image_name in annotations.keys():
            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            image = Image(image_path)
            patch_records = ep.extract_patch_records_from_image_tiled(
                image, 
                patch_size,
                image_annotations=None,
                patch_overlap_percent=50, 
                include_patch_arrays=True)

            ep.write_patches(patches_dir, patch_records)

            patch_records = ep.extract_patch_records_from_image_tiled(
                image, 
                patch_size,
                image_annotations=None,
                patch_overlap_percent=50, 
                include_patch_arrays=False)

            patch_data[image_name] = patch_records



        
        json_io.save_json(patch_data_path, patch_data)
    else:
        print("do not need to extract patches")


def handle_prediction_request(request):

    farm_name = request["farm_name"]
    field_name = request["field_name"]
    mission_date = request["mission_date"]


    status_path = os.path.join("usr", "data", "image_sets",
                                farm_name, field_name, mission_date, 
                                "model", "status.json")
    status = json_io.load_json(status_path)

    status["status"] = PREDICTING
    status["update_num"] = status["update_num"] + 1
    json_io.save_json(status_path, status)
    notify(farm_name, field_name, mission_date)

    predict_on_image(
        farm_name,
        field_name,
        mission_date,
        request["image_name"]
    )


    status = json_io.load_json(status_path)
    status["status"] = IDLE
    status["update_num"] = status["update_num"] + 1
    json_io.save_json(status_path, status)
    notify(farm_name, field_name, mission_date, extra_items={"prediction_image_name": request["image_name"]})



def check_train(farm_name, field_name, mission_date):

    
    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    training_dir = os.path.join(model_dir, "training")

    loss_record_path = os.path.join(training_dir, "loss_record.json")


    loss_record = json_io.load_json(loss_record_path)
    needs_training_with_cur_set = loss_record["validation_loss"]["epochs_since_improvement"] < yolov4_image_set_driver.VALIDATION_IMPROVEMENT_TOLERANCE


    status_path = os.path.join(model_dir, "status.json")

    status = json_io.load_json(status_path)

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    num_available = 0
    for image_name in annotations.keys():
        if annotations[image_name]["status"] == "completed_for_training":
            num_available += 1

    needs_training_with_new_set = num_available > loss_record["num_training_images"]

    if needs_training_with_cur_set or needs_training_with_new_set:

        status_path = os.path.join("usr", "data", "image_sets",
                                   farm_name, field_name, mission_date, 
                                   "model", "status.json")
        status = json_io.load_json(status_path)
        status["status"] = TRAINING
        status["update_num"] = status["update_num"] + 1

        json_io.save_json(status_path, status)

        notify(farm_name, field_name, mission_date)

        create_patches_if_needed(farm_name, field_name, mission_date)

        if needs_training_with_new_set:
            num_available = update_training_tf_record(farm_name, field_name, mission_date)

            loss_record = {
                "training_loss": { "values": [],
                                   "best": 100000000,
                                   "epochs_since_improvement": 0}, 
                "validation_loss": {"values": [],
                                    "best": 100000000,
                                    "epochs_since_improvement": 0},
                "num_training_images": num_available
            }

            json_io.save_json(loss_record_path, loss_record)

        training_finished = yolov4_image_set_driver.train(farm_name, field_name, mission_date)

        status = json_io.load_json(status_path)
        status["status"] = IDLE
        status["update_num"] = status["update_num"] + 1
        if training_finished:
            status["num_images_fully_trained_on"] = loss_record["num_training_images"]

        json_io.save_json(status_path, status)

        notify(farm_name, field_name, mission_date)




def predict_on_image(farm_name, field_name, mission_date, image_name):

    # saved_status = status["status"]
    # status["status"] = PREDICTING
    result = {
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
    }

    create_patches_if_needed(farm_name, field_name, mission_date)

    #image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)

    #dirty_path = os.path.join(image_set_dir, "annotations", "dirty.json")
    #dirty = json_io.load_json(dirty_path)
    #if dirty["dirty"]:

    print("updating prediction tf record...")
    update_prediction_tf_records(farm_name, field_name, mission_date, image_names=[image_name])
    print("finished.")


    
    yolov4_image_set_driver.predict(farm_name, field_name, mission_date, image_names=[image_name])

    return result

    # status["status"] = saved_status


def update_training_tf_record(farm_name, field_name, mission_date):

    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    patches_dir = os.path.join(image_set_dir, "patches")

    patch_data_path = os.path.join(patches_dir, "patch_data.json")
    patch_data = json_io.load_json(patch_data_path)

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    # available_for_training = []
    # for image_name in annotations.keys():
    #     if annotations[image_name]["available_for_training"]:
    #         available_for_training.append(image_name)

    training_dir = os.path.join(image_set_dir, "model", "training")
    #training_images_path = os.path.join(training_dir, "training_images.json")
    #cur_training_images = json_io.load_json(training_images_path)

    #if len(available_for_training) != len(cur_training_images):



    patch_records = []
    num_available = 0
    for image_name in annotations.keys():
        if annotations[image_name]["status"] == "completed_for_training":
            num_available += 1
            ep.add_annotations_to_patch_records(patch_data[image_name], annotations[image_name])
            patch_records.extend(patch_data[image_name])


    patch_records = np.array(patch_records)

    training_size = round(patch_records.size * 0.8)
    training_subset = random.sample(np.arange(patch_records.size).tolist(), training_size)

    training_patch_records = patch_records[training_subset]
    validation_patch_records = np.delete(patch_records, training_subset)


    training_tf_records = tf_record_io.create_patch_tf_records(training_patch_records, patches_dir, is_annotated=True)
    training_patches_record_path = os.path.join(training_dir, "training-patches-record.tfrec")
    tf_record_io.output_patch_tf_records(training_patches_record_path, training_tf_records)

    validation_tf_records = tf_record_io.create_patch_tf_records(validation_patch_records, patches_dir, is_annotated=True)
    validation_patches_record_path = os.path.join(training_dir, "validation-patches-record.tfrec")
    tf_record_io.output_patch_tf_records(validation_patches_record_path, validation_tf_records)

    return num_available




def update_prediction_tf_records(farm_name, field_name, mission_date, image_names):

    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    patches_dir = os.path.join(image_set_dir, "patches")

    patch_data_path = os.path.join(patches_dir, "patch_data.json")
    patch_data = json_io.load_json(patch_data_path)

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    for image_name in image_names:
        #is_annotated = annotations[image_name]["status"] == "completed"

        patch_records = patch_data[image_name]
        #if is_annotated:
        #    ep.add_annotations_to_patch_records(patch_data[image_name], annotations[image_name])


        print("sample patch record:", patch_records[0])


        image_prediction_dir = os.path.join(image_set_dir, "model", "prediction", "images", image_name)
        os.makedirs(image_prediction_dir, exist_ok=True)

        tf_records = tf_record_io.create_patch_tf_records(patch_records, patches_dir, is_annotated=False) #is_annotated)
        patches_record_path = os.path.join(image_prediction_dir, "patches-record.tfrec")
        tf_record_io.output_patch_tf_records(patches_record_path, tf_records)







def run(farm_name, field_name, mission_date, predict_on):

    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")

    # status_path = os.path.join(model_dir, "status.json")
    # model_status = json_io.load_json(status_path)

    if predict_on is not None:
        predict_on_image(farm_name, field_name, mission_date, predict_on)
    #request_path = os.path.join(model_dir, "request.json")
    #request = json_io.load_json(request_path)


    #if request["request"] == "predict_on_image":
    #    predict_on_image()


    # if model_status["status"] == EXTRACT_PATCHES:
    #     extract_patches(farm_name, field_name, mission_date)

    # elif model_status["status"] == TRAINING:
    #     train(farm_name, field_name, mission_date)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    #parser.add_argument('-predict', action='store_true')
    parser.add_argument("--predict_on", type=str)

    args = parser.parse_args()

    run(args.farm_name,
        args.field_name, 
        args.mission_date,
        predict_on=args.predict_on)

    print("now exiting")
    exit()