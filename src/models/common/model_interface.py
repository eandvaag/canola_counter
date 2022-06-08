from abc import ABC, abstractmethod
import os
import shutil
import random
#import randomname
import logging
import uuid

from io_utils import json_io

from models.common import model_keys

import extract_patches as ep
import build_datasets

#from models.retinanet import retinanet_driver
#from models.centernet import centernet_driver
#from models.efficientdet import efficientdet_driver
from models.yolov4 import yolov4_driver


class DetectorWrapper(ABC):

    def __init__(self, model_config):
        #self.config = self.custom_config(model_dir)
        model_keys.add_dir_shortcuts(model_config)
        model_keys.add_general_keys(model_config)
        model_keys.add_specialized_keys(model_config)
        self.config = model_config
        #self.config.add_arch_config(arch_config)

    def train_model(self):
        #self.config.add_training_config(train_config)
        build_datasets.build_training_datasets(self.config)
        self.custom_train_model()

    def run_inference(self):
        #self.config.add_inference_config(inference_config)
        build_datasets.build_inference_datasets(self.config)
        return self.custom_run_inference()



class RetinaNetWrapper(DetectorWrapper):

    def custom_train_model(self):
        retinanet_driver.train(self.config)

    def custom_run_inference(self):
        return retinanet_driver.generate_predictions(self.config)


class CenterNetWrapper(DetectorWrapper):

    def custom_train_model(self):
        centernet_driver.train(self.config)

    def custom_run_inference(self):
        return centernet_driver.generate_predictions(self.config)


class EfficientDetWrapper(DetectorWrapper):

    def custom_train_model(self):
        efficientdet_driver.train(self.config)

    def custom_run_inference(self):
        return efficientdet_driver.generate_predictions(self.config)


class YOLOv4Wrapper(DetectorWrapper):

    def custom_train_model(self):
        yolov4_driver.train(self.config)

    def custom_run_inference(self):
        return yolov4_driver.generate_predictions(self.config)    


def create_model_wrapper(model_config):

    model_type = model_config["arch"]["model_type"]

    if model_type == "retinanet":
        model_wrapper = RetinaNetWrapper(model_config)
    elif model_type == "centernet":
        model_wrapper = CenterNetWrapper(model_config)
    elif model_type == "efficientdet":
        model_wrapper = EfficientDetWrapper(model_config)
    elif model_type == "yolov4" or model_type == "yolov4_tiny":
        model_wrapper = YOLOv4Wrapper(model_config)
    else:
        raise RuntimeError("Unknown model type: '{}'.".format(model_type))

    return model_wrapper



# def handle_existing_model_data(model_uuid, model_name, on_found):

#     model_dir = os.path.join("usr", "data", "models", model_uuid)

#     arch_data = [model_dir, os.path.join(model_dir, "model_config.json")]

#     for item in arch_data:
#         if os.path.exists(item):
#             if on_found == "raise":
#                 raise RuntimeError("Existing model data found for '{}' (uuid: {})".format(
#                                    model_name, model_uuid))
#             elif on_found == "replace":
#                 if os.path.isdir(item):
#                     shutil.rmtree(item)
#                 else:
#                     os.remove(item)



def create_model(model_config, on_found="replace"):
    """
        on_found:
            "raise": raise a RuntimeError
            "replace": replace existing
    """

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")

    model_uuid = model_config["model_uuid"]
    model_name = model_config["model_name"]
    model_dir = os.path.join(usr_data_root, "models", model_uuid)
    arch_config_path = os.path.join(model_dir, "model_config.json")

    #handle_existing_model_data(model_uuid, model_name, on_found)
    if os.path.exists(model_dir):
        if on_found == "replace":
            shutil.rmtree(model_dir)
        else:
            raise RuntimeError("Model directory {} already exists.".format(model_uuid))

    os.makedirs(model_dir)
    
    json_io.save_json(arch_config_path, model_config)

    logger.info("Instantiated new model: '{}' (uuid: {})".format(model_name, model_uuid))




def handle_existing_training_data(model_uuid, model_name, on_found):

    model_dir = os.path.join("usr", "data", "models", model_uuid)

    training_data = [
        os.path.join(model_dir, "weights"),
        os.path.join(model_dir, "loss_records"),
        os.path.join(model_dir, "training_config.json"),
        #os.path.join(model_dir, "class_map.json"),
        os.path.join(model_dir, "training_patches")
    ]

    for item in training_data:
        if os.path.exists(item):
            if on_found == "raise":
                raise RuntimeError("Existing training data found for '{}' (uuid: {})".format(
                                   model_name, model_uuid))
            if on_found == "replace":
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)



def train_model(model_uuid, on_found="replace"):

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")
    model_dir = os.path.join(usr_data_root, "models", model_uuid)
    model_config_path = os.path.join(model_dir, "model_config.json")
    model_config = json_io.load_json(model_config_path)

    model_name = model_config["model_name"]

    handle_existing_training_data(model_uuid, model_name, on_found)

    logger.info("Started training model: '{}' (uuid: {})".format(model_name, model_uuid))
    
    
    weights_dir = os.path.join(model_dir, "weights")
    os.makedirs(weights_dir)
    loss_records_dir = os.path.join(model_dir, "loss_records")
    os.makedirs(loss_records_dir)

    #training_patch_dir, validation_patch_dir = 
    #ep.create_source_patches(training_config)
    #req_args["training_sequence"][0]["training_patch_dir"] = training_patch_dir
    #req_args["training_sequence"][0]["validation_patch_dir"] = validation_patch_dir


    #training_config_path = os.path.join(model_dir, "training_config.json")
    #json_io.save_json(training_config_path, training_config)


    model_wrapper = create_model_wrapper(model_config)

    model_wrapper.train_model()

    logger.info("Finished training model: '{}' (uuid: {})".format(model_name, model_uuid))
    



def handle_existing_inference_data(model_uuid, model_name, on_found):

    model_dir = os.path.join("usr", "data", "models", model_uuid)

    inference_data = [
        os.path.join(model_dir, "predictions"),
        os.path.join(model_dir, "inference_patches")
    ]

    for item in inference_data:
        if os.path.exists(item):
            if on_found == "raise":
                raise RuntimeError("Existing inference data found for '{}' (uuid: {})".format(
                                       model_name, model_uuid))
            if on_found == "replace":
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)


def run_inference(model_uuid, on_found="replace"):

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")
    model_dir = os.path.join(usr_data_root, "models", model_uuid)
    model_config_path = os.path.join(model_dir, "model_config.json")
    model_config = json_io.load_json(model_config_path)

    model_name = model_config["model_name"]

    handle_existing_inference_data(model_uuid, model_name, on_found)

    logger.info("Started running inference with: '{}' (uuid: {})".format(model_name, model_uuid))

    model_dir = os.path.join(usr_data_root, "models", model_uuid)
    #predictions_dir = os.path.join(model_dir, "predictions")
    #os.makedirs(predictions_dir)

    #inference_patch_dir = ep.create_target_patches(inference_config)
    #inference_config["inference_patch_dir"] = inference_patch_dir

    #inference_config_path = os.path.join(model_dir, "inference_config.json")
    #json_io.save_json(inference_config_path, inference_config)

    model_wrapper = create_model_wrapper(model_config)

    model_wrapper.run_inference()

    logger.info("Finished running inference with: '{}' (uuid: {})".format(model_name, model_uuid))