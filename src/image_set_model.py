import logging
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
from image_set_actions import notify, INITIALIZING, IDLE, PREDICTING, TRAINING
#import image_set_actions as isa


MIN_NUM_ANNOTATIONS_BASELINE_EVAL = 20
MIN_NUM_ANNOTATIONS_BASELINE_CREATE = 1000



def create_patches_if_needed(farm_name, field_name, mission_date, image_names):
    
    logger = logging.getLogger(__name__)

    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    images_dir = os.path.join(image_set_dir, "images")
    patches_dir = os.path.join(image_set_dir, "patches")
    patch_data_path = os.path.join(patches_dir, "patch_data.json")

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    try:
        updated_patch_size = w3c_io.get_patch_size(annotations)
    except RuntimeError:
        updated_patch_size = 300
    logger.info("Updated patch size: {}".format(updated_patch_size))


    # if os.path.exists(patch_data_path):
    #     existing_patch_size = imagesize.get(glob.glob(os.path.join(patches_dir, "*.png"))[0])[0]
    #     print("existing patch size", existing_patch_size)
    #     abs_patch_size_diff = abs(existing_patch_size - patch_size)

    update_thresh = 10

    if os.path.exists(patch_data_path):
        patch_data = json_io.load_json(patch_data_path)
    else:
        patch_data = {}
    

    for image_name in image_names:

        needs_update = True
        if image_name in patch_data:
            sample_patch_coords = patch_data[image_name][0]["patch_coords"]
            existing_patch_size = sample_patch_coords[2] - sample_patch_coords[0]
            logger.info("Existing patch size: {}".format(existing_patch_size))
            abs_patch_size_diff = abs(existing_patch_size - updated_patch_size)
            needs_update = abs_patch_size_diff >= update_thresh

        if needs_update:
            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            image = Image(image_path)
            patch_records = ep.extract_patch_records_from_image_tiled(
                image, 
                updated_patch_size,
                image_annotations=None,
                patch_overlap_percent=50, 
                include_patch_arrays=True)

            ep.write_patches(patches_dir, patch_records)

            patch_records = ep.extract_patch_records_from_image_tiled(
                image, 
                updated_patch_size,
                image_annotations=None,
                patch_overlap_percent=50, 
                include_patch_arrays=False)

            patch_data[image_name] = patch_records
            #patch_data[image_name]["records"] = patch_records
            #patch_data[image_name]["status"] = annotations[image_name]["status"]
        
        json_io.save_json(patch_data_path, patch_data)



def handle_resume_direct_baseline_request(baseline_name):
    logger = logging.getLogger(__name__)
    logger.info("Resuming baseline model '{}'".format(baseline_name))

    baseline_dir = os.path.join("usr", "data", "baselines", baseline_name)
    model_dir = os.path.join(baseline_dir, "model")
    weights_dir = os.path.join(model_dir, "weights")
    assert os.path.exists(baseline_dir)

    yolov4_image_set_driver.train(baseline_dir)

    shutil.move(os.path.join(weights_dir, "best_weights.h5"),
                os.path.join("usr", "data", "baselines", baseline_name + ".h5"))
    shutil.rmtree(baseline_dir)


def handle_direct_baseline_request(request):

    baseline_name = request["baseline_name"]

    baseline_dir = os.path.join("usr", "data", "baselines", baseline_name) #"training", baseline_name)
    patches_dir = os.path.join(baseline_dir, "patches")
    model_dir = os.path.join(baseline_dir, "model")
    training_dir = os.path.join(model_dir, "training")
    weights_dir = os.path.join(model_dir, "weights")

    if os.path.exists(baseline_dir):
        raise RuntimeError("Baseline directory already exists!")

    # if not os.path.exists(baseline_dir):
    os.makedirs(baseline_dir)
    # if not os.path.exists(patches_dir):
    os.makedirs(patches_dir)
    # if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    # if not os.path.exists(training_dir):
    os.makedirs(training_dir)
    # if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)


    all_records = []
    for image_set in request["image_sets"]:
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", "image_sets", 
                                     farm_name, field_name, mission_date)
        images_dir = os.path.join(image_set_dir, "images")

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

        patch_size = w3c_io.get_patch_size(annotations)

        #for image_name in request["image_sets"][image_set]["images"]:
        for image_name in image_set["images"]:


            image_path = glob.glob(os.path.join(images_dir, (image_name) + ".*"))[0]
            image = Image(image_path)
            patch_records = ep.extract_patch_records_from_image_tiled(
                image, 
                patch_size,
                image_annotations=annotations[image_name],
                patch_overlap_percent=50, 
                include_patch_arrays=True)

            ep.write_patches(patches_dir, patch_records)
            all_records.extend(patch_records)

    patch_records = np.array(all_records)

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

    loss_record_path = os.path.join(training_dir, "loss_record.json")
    num_images = int(np.sum([len(image_set["images"]) for image_set in request["image_sets"]]))
    loss_record = {
        "training_loss": { "values": [],
                            "best": 100000000,
                            "epochs_since_improvement": 0}, 
        "validation_loss": {"values": [],
                            "best": 100000000,
                            "epochs_since_improvement": 0},
        "num_training_images": num_images
    }

    json_io.save_json(loss_record_path, loss_record)



    yolov4_image_set_driver.train(baseline_dir)

    shutil.move(os.path.join(weights_dir, "best_weights.h5"),
                os.path.join("usr", "data", "baselines", baseline_name + ".h5"))
    shutil.rmtree(baseline_dir)
    # if completed:
    #     shutil.move("usr/data/baselines/training/" + baseline_name,
    #                 "usr/data/baselines/completed/" + baseline_name)



def handle_prediction_request(request):

    farm_name = request["farm_name"]
    field_name = request["field_name"]
    mission_date = request["mission_date"]

    #if check_baseline(farm_name, field_name, mission_date):

    status_path = os.path.join("usr", "data", "image_sets",
                                farm_name, field_name, mission_date, 
                                "model", "status.json")
    status = json_io.load_json(status_path)

    status["status"] = PREDICTING
    status["update_num"] = status["update_num"] + 1
    json_io.save_json(status_path, status)
    notify(farm_name, field_name, mission_date)


    save_result = request["save_result"]

    predict_on_images(
        farm_name,
        field_name,
        mission_date,
        request["image_names"],
        save_result
    )


    status = json_io.load_json(status_path)
    status["status"] = IDLE
    status["update_num"] = status["update_num"] + 1
    json_io.save_json(status_path, status)
    notify(farm_name, field_name, mission_date, extra_items={"prediction_image_names": ",".join(request["image_names"])})




def check_train(farm_name, field_name, mission_date):

    #baseline_exists = check_baseline(farm_name, field_name, mission_date)
    #if baseline_exists:
    
    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    training_dir = os.path.join(model_dir, "training")
    weights_dir = os.path.join(model_dir, "weights")

    loss_record_path = os.path.join(training_dir, "loss_record.json")


    loss_record = json_io.load_json(loss_record_path)
    needs_training_with_cur_set = loss_record["validation_loss"]["epochs_since_improvement"] < yolov4_image_set_driver.VALIDATION_IMPROVEMENT_TOLERANCE


    status_path = os.path.join(model_dir, "status.json")

    status = json_io.load_json(status_path)

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
    num_training_annotations = w3c_io.get_num_annotations(annotations, require_completed_for_training=True)

    #num_available = 0
    training_image_names = []
    for image_name in annotations.keys():
        if annotations[image_name]["status"] == "completed_for_training":
            training_image_names.append(image_name)
            #num_available += 1

    needs_training_with_new_set = len(training_image_names) > loss_record["num_training_images"]

    if needs_training_with_cur_set or needs_training_with_new_set:

        status_path = os.path.join("usr", "data", "image_sets",
                                farm_name, field_name, mission_date, 
                                "model", "status.json")
        status = json_io.load_json(status_path)
        status["status"] = TRAINING
        status["update_num"] = status["update_num"] + 1

        json_io.save_json(status_path, status)

        notify(farm_name, field_name, mission_date)

        create_patches_if_needed(farm_name, field_name, mission_date, training_image_names)

        if needs_training_with_new_set:
            update_training_tf_record(farm_name, field_name, mission_date, training_image_names)

            loss_record = {
                "training_loss": { "values": [],
                                "best": 100000000,
                                "epochs_since_improvement": 0}, 
                "validation_loss": {"values": [],
                                    "best": 100000000,
                                    "epochs_since_improvement": 0},
                "num_training_images": len(training_image_names)
            }

            json_io.save_json(loss_record_path, loss_record)

        training_finished = yolov4_image_set_driver.train(image_set_dir) #farm_name, field_name, mission_date)

        status = json_io.load_json(status_path)
        status["status"] = IDLE
        status["update_num"] = status["update_num"] + 1
        if training_finished:
            status["num_images_fully_trained_on"] = loss_record["num_training_images"]


            if num_training_annotations >= MIN_NUM_ANNOTATIONS_BASELINE_CREATE:
                baseline_name = farm_name + "::" + field_name + "::" + mission_date
                shutil.copyfile(os.path.join(weights_dir, "best_weights.h5"),
                            os.path.join("usr", "data", "baselines", baseline_name + ".h5"))

        json_io.save_json(status_path, status)

        notify(farm_name, field_name, mission_date)




def predict_on_images(farm_name, field_name, mission_date, image_names, save_result):

    create_patches_if_needed(farm_name, field_name, mission_date, image_names)

    update_prediction_tf_records(farm_name, field_name, mission_date, image_names=image_names)
    
    yolov4_image_set_driver.predict(farm_name, field_name, mission_date, image_names=image_names, save_result=save_result)





def update_training_tf_record(farm_name, field_name, mission_date, training_image_names):

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
    # num_available = 0
    # for image_name in annotations.keys():
    #     if annotations[image_name]["status"] == "completed_for_training":
    #         num_available += 1

    for image_name in training_image_names:
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

    # return num_available




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



#def determine_best_baseline(farm_name, field_name, mission_date):
def handle_baseline_request(request): #farm_name, field_name, mission_date):

    logger = logging.getLogger(__name__)


    farm_name = request["farm_name"]
    field_name = request["field_name"]
    mission_date = request["mission_date"]
    annotation_guides = request["annotation_guides"]


    status_path = os.path.join("usr", "data", "image_sets",
                                farm_name, field_name, mission_date, 
                                "model", "status.json")
    status = json_io.load_json(status_path)

    status["status"] = INITIALIZING
    status["update_num"] = status["update_num"] + 1
    json_io.save_json(status_path, status)
    notify(farm_name, field_name, mission_date)


    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    best_weights_path = os.path.join(model_dir, "weights", "best_weights.h5")



    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations_json = json_io.load_json(annotations_path)
    annotations = w3c_io.convert_json_annotations(annotations_json, {"plant": 0})
    num_annotations = w3c_io.get_num_annotations(annotations, require_completed_for_training=False)

    # if num_annotations < MIN_NUM_ANNOTATIONS_BASELINE_EVAL:
    #     return False

    # if os.path.exists(best_weights_path):
    #     return True
    
    logger.info("No weights found. Selecting initial weights...")

    images_dir = os.path.join(image_set_dir, "images")
    #weights_dir = os.path.join(model_dir, "weights")



    init_dir = os.path.join(model_dir, "init")
    os.makedirs(init_dir, exist_ok=True)
    init_patches_dir = os.path.join(init_dir, "patches")
    os.makedirs(init_patches_dir, exist_ok=True)

    patch_size = w3c_io.get_patch_size(annotations)
    patch_records = []
    for image_name in annotations.keys():
        if len(annotation_guides[image_name]) > 0:
            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            image = Image(image_path)
            patch_records.extend(ep.extract_patches_from_annotation_guides(image, patch_size, annotations[image_name], annotation_guides[image_name]))

            # image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            # image = Image(image_path)

            # patch_records.extend(ep.extract_patch_records_surrounding_annotations(image, patch_size, annotations[image_name]))
        
    # augmentations = [
    #     {
    #         "type": "flip_vertical", 
    #         "parameters": {
    #             "probability": 0.5
    #         }
    #     },
    #     {
    #         "type": "flip_horizontal", 
    #         "parameters": {
    #             "probability": 0.5
    #         }
    #     },
    #     {
    #         "type": "rotate_90", 
    #         "parameters": {
    #             "probability": 0.5
    #         }
    #     }
    # ]

    # from models.common import data_augment
    # aug_patch_records = []
    # for patch_record in patch_records:
    #     for augmentation in augmentations:
    #         patch, boxes, classes = data_augment.apply_augmentations([augmentation], 
    #         patch_record["patch"], patch_record["patch_normalized_boxes"], patch_record["patch_classes"])
    #         aug_patch_records.append({

    #             "patch": patch,
    #             "patch_name": augmentation["type"] + "_" + patch_record["patch_name"],
    #             "image_path": patch_record["image_path"],
    #             "image_name": patch_record["image_name"],
    #             "patch_coords": patch_record["patch_coords"],
    #             "image_abs_boxes": patch_record["image_abs_boxes"], # FAKE
    #             "patch_normalized_boxes": boxes.tolist(),
    #             "patch_abs_boxes": np.rint(
    #                     np.stack([
    #                         boxes[..., 0] * patch.shape[1],
    #                         boxes[..., 1] * patch.shape[0],
    #                         boxes[..., 2] * patch.shape[1],
    #                         boxes[..., 3] * patch.shape[0],

    #                     ], axis=-1)
    #             ).tolist(),
    #             "patch_classes": classes.tolist()
    #         })
    # patch_records.extend(aug_patch_records)
    
    logger.info("Writing initial patches")
    ep.write_patches(init_patches_dir, patch_records)
    #print(patch_records)

    tf_records = tf_record_io.create_patch_tf_records(patch_records, init_patches_dir, is_annotated=True) #False) #is_annotated)
    patches_record_path = os.path.join(init_dir, "patches-record.tfrec")
    tf_record_io.output_patch_tf_records(patches_record_path, tf_records)


    yolov4_image_set_driver.select_baseline(farm_name, field_name, mission_date)


    logger.info("Initial weights have been chosen.")


    status = json_io.load_json(status_path)

    status["status"] = IDLE
    status["update_num"] = status["update_num"] + 1
    json_io.save_json(status_path, status)
    notify(farm_name, field_name, mission_date)

    # return True