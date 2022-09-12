import logging
import os
import shutil
import glob

from io_utils import json_io

import excess_green
import metadata


init_cameras = {
    "GoPro": {
        "HERO9 Black": {
            "sensor_width": 6.17,
            "sensor_height": 4.55,
            "focal_length": 3
        },
        "HERO6 Black": {
            "sensor_width": 6.17,
            "sensor_height": 4.55,
            "focal_length": 3
        }
    },
    "Hasselblad": {
        "L1D-20c": {
            "sensor_width": 13.2,
            "sensor_height": 8.8,
            "focal_length": 10.3
        }
    },
    "Phase One": {
        "iXU1000": {
            "sensor_width": 53.4,
            "sensor_height": 40.0,
            "focal_length": 55.0
        }
    }
}

def clear_usr_requests_and_results(username):

    usr_dir = os.path.join("usr", "data", username)
    for farm_path in glob.glob(os.path.join(usr_dir, "image_sets", "*")):
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                prediction_dir = os.path.join(mission_path, "model", "prediction")
                image_requests_dir = os.path.join(prediction_dir, "image_requests")
                shutil.rmtree(image_requests_dir)
                os.makedirs(image_requests_dir)

                images_dir = os.path.join(prediction_dir, "images")
                if os.path.exists(images_dir):
                    shutil.rmtree(images_dir)
                    os.makedirs(images_dir)

                image_set_requests_dir = os.path.join(prediction_dir, "image_set_requests")
                aborted_dir = os.path.join(image_set_requests_dir, "aborted")
                shutil.rmtree(aborted_dir)
                os.makedirs(aborted_dir)

                pending_dir = os.path.join(image_set_requests_dir, "pending")
                shutil.rmtree(pending_dir)
                os.makedirs(pending_dir)                

                results_dir = os.path.join(mission_path, "model", "results")
                shutil.rmtree(results_dir)
                os.makedirs(results_dir)



def init_usr(username):

    usr_dir = os.path.join("usr", "data", username)
    cameras_dir = os.path.join(usr_dir, "cameras")
    image_sets_dir = os.path.join(usr_dir, "image_sets")


    os.makedirs(usr_dir)
    os.makedirs(cameras_dir)
    os.makedirs(image_sets_dir)

    cameras_path = os.path.join(cameras_dir, "cameras.json")
    json_io.save_json(cameras_path, init_cameras)


def update_init_cameras(replace=False):

    for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
        cameras_path = os.path.join(usr_dir, "cameras", "cameras.json")

        if os.path.exists(cameras_path):

            if replace:
                json_io.save_json(cameras_path, init_cameras)

            else:
                usr_cameras = json_io.load_json(cameras_path)
                for make in init_cameras.keys():
                    if make not in usr_cameras:
                        usr_cameras[make] = {}
                    for model in init_cameras[make].keys():
                        usr_cameras[make][model] = init_cameras[make][model]

                json_io.save_json(cameras_path, usr_cameras)



def fix_k():

    for farm_path in glob.glob(os.path.join("usr", "data", "kaylie", "image_sets", "*")):
        farm_dir = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_dir = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_dir = os.path.basename(mission_path)

                annotations_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                annotations = json_io.load_json(annotations_path)
                fully_trained = "True"
                for image_name in annotations.keys():
                    annotations[image_name]["update_time"] = 0 #int(time.time())
                    if annotations[image_name]["status"] == "completed_for_training":
                        fully_trained = "False"

                json_io.save_json(annotations_path, annotations)

                # status = {
                #     "status": "idle",
                #     "fully_trained": "True",
                #     "update_num": 0
                # }
                status_path = os.path.join(mission_path, "model", "status.json")
                status = json_io.load_json(status_path)
                status["fully_trained"] = fully_trained

                json_io.save_json(status_path, status)

def fix_kaylie_sets():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    kaylie_image_sets_dir = os.path.join("usr", "data", "kaylie", "image_sets")

    #usr_dir = os.path.join("usr", "data", username)
    for farm_path in glob.glob(os.path.join("usr", "data", "image_sets", "*")):
        farm_dir = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_dir = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_dir = os.path.basename(mission_path)

                if (farm_dir == "row_spacing" and field_dir == "brown") and mission_dir == "2021-06-01-low_res":
                    pass
                else:

                    print("processing: {} {} {}".format(farm_dir, field_dir, mission_dir))

                    new_farm_path = os.path.join(kaylie_image_sets_dir, farm_dir)
                    new_field_path = os.path.join(new_farm_path, field_dir)
                    new_mission_path = os.path.join(new_field_path, mission_dir)

                    upload_status_path = os.path.join(new_mission_path, "upload_status.json")
                    if not os.path.exists(upload_status_path):

                        if not os.path.exists(new_farm_path):
                            os.makedirs(new_farm_path)
                        
                        if not os.path.exists(new_field_path):
                            os.makedirs(new_field_path)
                        
                        if not os.path.exists(new_mission_path):
                            os.makedirs(new_mission_path)

                        logger.info("Copying images")
                        images_path = os.path.join(mission_path, "images")
                        new_images_path = os.path.join(new_mission_path, "images")
                        # print(images_path)
                        # print(new_images_path)
                        shutil.copytree(images_path, new_images_path)

                        logger.info("Copying DZI images")
                        dzi_images_path = os.path.join(mission_path, "dzi_images")
                        new_dzi_images_path = os.path.join(new_mission_path, "dzi_images")
                        shutil.copytree(dzi_images_path, new_dzi_images_path)

                        logger.info("Copying annotations")
                        new_annotations_dir = os.path.join(new_mission_path, "annotations")
                        os.makedirs(new_annotations_dir)

                        old_annotations_w3c_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                        annotations_w3c_path = os.path.join(new_annotations_dir, "annotations_w3c.json")
                        shutil.copy(old_annotations_w3c_path, annotations_w3c_path)

                        # annotations_path = os.path.join(mission_path, "annotations")
                        # new_annotations_path = os.path.join(new_mission_path, "annotations")
                        # shutil.copytree(annotations_path, new_annotations_path)
                        
                        annotations = json_io.load_json(annotations_w3c_path)
                        fully_trained = "True"
                        for image_name in annotations.keys():
                            annotations[image_name]["update_time"] = 0 #int(time.time())
                            if annotations[image_name]["status"] == "completed":
                                if mission_dir[:4] == "2022":
                                    annotations[image_name]["status"] = "completed_for_testing"
                                else:
                                    annotations[image_name]["status"] = "completed_for_training"
                                    fully_trained = "False"

                                

                        json_io.save_json(annotations_w3c_path, annotations)


                        logger.info("Creating directories")
                        patches_dir = os.path.join(new_mission_path, "patches")
                        os.makedirs(patches_dir)

                        model_dir = os.path.join(new_mission_path, "model")
                        os.makedirs(model_dir)
                        prediction_dir = os.path.join(model_dir, "prediction")
                        os.makedirs(prediction_dir)
                        image_requests_dir = os.path.join(prediction_dir, "image_requests")
                        os.makedirs(image_requests_dir)
                        images_dir = os.path.join(prediction_dir, "images")
                        os.makedirs(images_dir)
                        image_set_requests_dir = os.path.join(prediction_dir, "image_set_requests")
                        os.makedirs(image_set_requests_dir)
                        aborted_dir = os.path.join(image_set_requests_dir, "aborted")
                        os.makedirs(aborted_dir)
                        pending_dir = os.path.join(image_set_requests_dir, "pending")
                        os.makedirs(pending_dir)

                        results_dir = os.path.join(model_dir, "results")
                        os.makedirs(results_dir)
                        # retrieval_dir = os.path.join(results_dir, "retrieval")
                        # os.makedirs(retrieval_dir)
                        training_dir = os.path.join(model_dir, "training")
                        os.makedirs(training_dir)
                        usr_block = {}
                        usr_block_path = os.path.join(training_dir, "usr_block.json")
                        json_io.save_json(usr_block_path, usr_block)

                        loss_record = {
                            "training_loss": { "values": [],
                                            "best": 100000000,
                                            "epochs_since_improvement": 100000000}, 
                            "validation_loss": {"values": [],
                                                "best": 100000000,
                                                "epochs_since_improvement": 100000000},
                        }
                        loss_record_path = os.path.join(training_dir, "loss_record.json")
                        json_io.save_json(loss_record_path, loss_record)

                        weights_dir = os.path.join(model_dir, "weights")
                        os.makedirs(weights_dir)

                        default_weights_path = os.path.join("usr", "shared", "weights", "default_weights.h5")
                        shutil.copy(default_weights_path, os.path.join(weights_dir, "best_weights.h5"))
                        shutil.copy(default_weights_path, os.path.join(weights_dir, "cur_weights.h5"))

                        status = {
                            "status": "idle",
                            "fully_trained": fully_trained,
                            "update_num": 0
                        }
                        status_path = os.path.join(model_dir, "status.json")
                        json_io.save_json(status_path, status)


                        logger.info("Creating excess green images")
                        excess_green_dir = os.path.join(new_mission_path, "excess_green")
                        os.makedirs(excess_green_dir)
                        excess_green.create_excess_green_for_image_set(new_mission_path)


                        logger.info("Extracting metadata")
                        metadata.extract_metadata(new_mission_path, camera_height=2.0)

                        
                        logger.info("Writing upload status")
                        upload_status = {"status": "uploaded"}
                        json_io.save_json(upload_status_path, upload_status)

