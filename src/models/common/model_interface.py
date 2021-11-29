from abc import ABC, abstractmethod
import os
import shutil
import random
import randomname
import logging
import uuid

from io_utils import json_io

from models.common import model_config

from models.retinanet import retinanet_driver
from models.centernet import centernet_driver
from models.efficientdet import efficientdet_driver
from models.yolov4 import yolov4_driver


class DetectorWrapper(ABC):

    def __init__(self, model_dir, arch_config):
        self.config = self.custom_config(model_dir)
        self.config.add_arch_config(arch_config)

    def train_model(self, train_config):
        self.config.add_training_config(train_config)
        self.custom_train_model()

    def run_inference(self, inference_config):
        self.config.add_inference_config(inference_config)
        return self.custom_run_inference()



class RetinaNetWrapper(DetectorWrapper):

    def custom_config(self, model_dir):
        return model_config.RetinaNetConfig(model_dir)

    def custom_train_model(self):
        retinanet_driver.train(self.config)

    def custom_run_inference(self):
        return retinanet_driver.generate_predictions(self.config)


class CenterNetWrapper(DetectorWrapper):

    def custom_config(self, model_dir):
        return model_config.CenterNetConfig(model_dir)

    def custom_train_model(self):
        centernet_driver.train(self.config)

    def custom_run_inference(self):
        return centernet_driver.generate_predictions(self.config)


class EfficientDetWrapper(DetectorWrapper):

    def custom_config(self, model_dir):
        return model_config.EfficientDetConfig(model_dir)

    def custom_train_model(self):
        efficientdet_driver.train(self.config)

    def custom_run_inference(self):
        return efficientdet_driver.generate_predictions(self.config)


class YOLOv4Wrapper(DetectorWrapper):

    def custom_config(self, model_dir):
        return model_config.YOLOv4Config(model_dir)

    def custom_train_model(self):
        yolov4_driver.train(self.config)

    def custom_run_inference(self):
        return yolov4_driver.generate_predictions(self.config)    


def create_model_wrapper(model_dir):

    arch_config_path = os.path.join(model_dir, "arch_config.json")

    arch_config = json_io.load_json(arch_config_path)

    model_type = arch_config["model_type"]

    if model_type == "retinanet":
        model_wrapper = RetinaNetWrapper(model_dir, arch_config)
    elif model_type == "centernet":
        model_wrapper = CenterNetWrapper(model_dir, arch_config)
    elif model_type == "efficientdet":
        model_wrapper = EfficientDetWrapper(model_dir, arch_config)
    elif model_type == "yolov4" or model_type == "yolov4_tiny":
        model_wrapper = YOLOv4Wrapper(model_dir, arch_config)
    else:
        raise RuntimeError("Unknown model type: '{}'.".format(model_type))

    return model_wrapper



def handle_existing_arch_data(model_uuid, model_name, on_found):

    model_dir = os.path.join("usr", "data", "models", model_uuid)

    arch_data = [model_dir, os.path.join(model_dir, "arch_config.json")]

    for item in arch_data:
        if os.path.exists(item):
            if on_found == "raise":
                raise RuntimeError("Existing training data found for '{}' (uuid: {})".format(
                                   model_name, model_uuid))
            elif on_found == "replace":
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)                



def create_model(req_args, on_found="raise"):
    """
        on_found:
            "raise": raise a RuntimeError
            "replace": replace existing
    """

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")

    if "model_uuid" not in req_args:
        req_args["model_uuid"] = str(uuid.uuid4())

    if "model_name" not in req_args:
        req_args["model_name"] = randomname.get_name()

    model_uuid = req_args["model_uuid"]
    model_name = req_args["model_name"]
    model_dir = os.path.join(usr_data_root, "models", model_uuid)
    arch_config_path = os.path.join(model_dir, "arch_config.json")

    handle_existing_arch_data(model_uuid, model_name, on_found)

    os.makedirs(model_dir)
    
    json_io.save_json(arch_config_path, req_args)

    logger.info("Instantiated new model: '{}' (uuid: {})".format(model_name, model_uuid))


def handle_existing_training_data(model_uuid, model_name, on_found):

    model_dir = os.path.join("usr", "data", "models", model_uuid)

    training_data = [
        os.path.join(model_dir, "weights"),
        os.path.join(model_dir, "loss_records"),
        os.path.join(model_dir, "training_config.json"),
        os.path.join(model_dir, "class_map.json")]

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

def train_model(req_args, on_found="raise"):

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")
    model_uuid = req_args["model_uuid"]
    model_name = req_args["model_name"]

    handle_existing_training_data(model_uuid, model_name, on_found)

    logger.info("Started training model: '{}' (uuid: {})".format(model_name, model_uuid))
    
    model_dir = os.path.join(usr_data_root, "models", model_uuid)
    weights_dir = os.path.join(model_dir, "weights")
    os.makedirs(weights_dir)
    loss_records_dir = os.path.join(model_dir, "loss_records")
    os.makedirs(loss_records_dir)


    train_config = req_args
    train_config_path = os.path.join(model_dir, "training_config.json")
    json_io.save_json(train_config_path, train_config)

    model_wrapper = create_model_wrapper(model_dir)

    model_wrapper.train_model(train_config)

    logger.info("Finished training model: '{}' (uuid: {})".format(model_name, model_uuid))
    



def handle_existing_inference_data(model_uuid, model_name, on_found):

    model_dir = os.path.join("usr", "data", "models", model_uuid)

    inference_data = [
        os.path.join(model_dir, "predictions")
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


def run_inference(req_args, on_found="raise"):

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")
    model_uuid = req_args["model_uuid"]
    model_name = req_args["model_name"]

    handle_existing_inference_data(model_uuid, model_name, on_found)

    logger.info("Started running inference with: '{}' (uuid: {})".format(model_name, model_uuid))

    model_dir = os.path.join(usr_data_root, "models", model_uuid)
    predictions_dir = os.path.join(model_dir, "predictions")
    os.makedirs(predictions_dir)

    inference_config = req_args
    inference_config_path = os.path.join(model_dir, "inference_config.json")
    json_io.save_json(inference_config_path, inference_config)

    model_wrapper = create_model_wrapper(model_dir)

    model_wrapper.run_inference(inference_config)

    logger.info("Finished running inference with: '{}' (uuid: {})".format(model_name, model_uuid))