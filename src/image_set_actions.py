import os
import glob
import shutil
import requests

from io_utils import json_io

#import image_set_model as ism




notification_url = "http://172.16.1.71:8110/plant_detection/notification"

INITIALIZING = "initializing"
IDLE = "idle"
TRAINING = "training"
PREDICTING = "predicting"



def notify(username, farm_name, field_name, mission_date, error=False, extra_items={}):
    #print("sending data", data)


    status_path = os.path.join("usr", "data", username, "image_sets",
                                farm_name, field_name, mission_date, 
                                "model", "status.json")
    status = json_io.load_json(status_path)

    data = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
        "error": error
    }
    for k, v in status.items():
        data[k] = v
    for k, v in extra_items.items():
        data[k] = v

    print(data)

    response = requests.post(notification_url, data=data)
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 200:
        print("Response status code is not 200. Status code: {}".format(response.status_code))
        response = response.json()
        print(response)

    response = response.json()
    if response["message"] != "received":
        print("Response message is not 'received'.")
        print(response)
        #exit()




def check_for_predictions():

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

                            model_dir = os.path.join("usr", "data", username, "image_sets",
                                                    farm_name, field_name, mission_date, 
                                                    "model")
                            prediction_dir = os.path.join(model_dir, "prediction")
                            prediction_requests_dirs = [
                                os.path.join(prediction_dir, "image_requests"),
                                os.path.join(prediction_dir, "image_set_requests", "pending")
                            ]

                            for prediction_requests_dir in prediction_requests_dirs:
                                prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
                                if len(prediction_request_paths) > 0:
                                    return True

    return False


# def check_restart(farm_name=None, field_name=None, mission_date=None):
#     restart_requests_dir = os.path.join("usr", "requests", "restart")

#     restart_request_paths = glob.glob(os.path.join(restart_requests_dir, "*"))

#     ret = False
#     while len(restart_request_paths) > 0:
#         restart_request_path = restart_request_paths[0]

#         restart_request = json_io.load_json(restart_request_path)

#         req_farm_name = restart_request["farm_name"]
#         req_field_name = restart_request["field_name"]
#         req_mission_date = restart_request["mission_date"]

#         if req_farm_name == farm_name and \
#            req_field_name == field_name and \
#            req_mission_date == mission_date:
#             ret = True

#         print("restarting {}::{}::{}".format(req_farm_name, req_field_name, req_mission_date))

#         image_set_dir = os.path.join("usr", "data", "image_sets",
#                                      req_farm_name, req_field_name, req_mission_date)

#         status_path = os.path.join(image_set_dir, "model", "status.json")
#         status = json_io.load_json(status_path)
#         status["status"] = IDLE
#         status["num_images_fully_trained_on"] = 0
#         status["update_num"] = status["update_num"] + 1
#         json_io.save_json(status_path, status)

#         baseline_weights_path = os.path.join("usr", "data", "weights", "default_weights.h5")
#         cur_weights_path = os.path.join(image_set_dir, "model", "weights", "cur_weights.h5")
#         best_weights_path = os.path.join(image_set_dir, "model", "weights", "best_weights.h5")
#         shutil.copyfile(baseline_weights_path, cur_weights_path)
#         shutil.copyfile(baseline_weights_path, best_weights_path)

#         prediction_images_dir = os.path.join(image_set_dir, "model", "prediction", "images")
#         if os.path.exists(prediction_images_dir):
#             shutil.rmtree(prediction_images_dir)
#             os.makedirs(prediction_images_dir)

#         annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
#         annotations = json_io.load_json(annotations_path)
#         for image_name in annotations.keys():
#             if annotations[image_name]["status"] == "completed_for_training":
#                 annotations[image_name]["status"] = "completed_for_testing"
#         json_io.save_json(annotations_path, annotations)


#         loss_record_path = os.path.join(image_set_dir, "model", "training", "loss_record.json")
#         loss_record = {
#                 "training_loss": { "values": [],
#                                    "best": 100000000,
#                                    "epochs_since_improvement": 100000000}, 
#                 "validation_loss": {"values": [],
#                                     "best": 100000000,
#                                     "epochs_since_improvement": 100000000},
#                 "num_training_images": 0
#         }
#         json_io.save_json(loss_record_path, loss_record)

#         notify(req_farm_name, req_field_name, req_mission_date, extra_items={"restarted": True})

#         os.remove(restart_request_path)
#         restart_request_paths = glob.glob(os.path.join(restart_requests_dir, "*"))

#     return ret


