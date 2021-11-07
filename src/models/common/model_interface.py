from abc import ABC, abstractmethod
import os
import shutil
import numpy as np
import random
import randomname
import logging
import copy
import uuid

from image_set import ImgSet
import extract_patches as ep
from io_utils import json_io

import models.common.model_config as model_config
import models.common.model_vis as model_vis
import models.common.decode_predictions as decode_predictions

from models.retinanet import retinanet_driver
from models.centernet import centernet_driver
from models.efficientdet import efficientdet_driver


class DetectorWrapper(ABC):

    def __init__(self, model_dir, arch_config):
        self.config = self.custom_config(model_dir)
        self.config.add_config(arch_config, "arch")

    def create_patches(self, extraction_params, img_set, dataset_name):
        dataset = img_set.datasets[dataset_name]

        scenario_id = ep.extract_patches(extraction_params,
                                         img_set,
                                         dataset,
                                         annotate_patches=dataset.is_annotated)

        patch_dir = os.path.join(img_set.patch_dir, scenario_id)
        return patch_dir, dataset.is_annotated


    def train_model(self, train_config):

        training_patch_dirs = []
        validation_patch_dirs = []
        self.config.add_config(train_config, "training")

        for img_set_conf in self.config.training_img_set_confs:
            img_set = ImgSet(img_set_conf["trial_name"], img_set_conf["mission_date"])
            training_patch_dir, _ = self.create_patches(self.config.training_patch_extraction_params, 
                                                          img_set, "training")
            validation_patch_dir, _ = self.create_patches(self.config.validation_patch_extraction_params, 
                                                            img_set, "validation")

            training_patch_dirs.append(training_patch_dir)
            validation_patch_dirs.append(validation_patch_dir)

        #json_io.save_json(os.path.join(self.config.model_dir, "training_config.json"), train_config)

        self.custom_train_model(training_patch_dirs, validation_patch_dirs)

    def run_inference(self, inference_config):

        logger = logging.getLogger(__name__)

        self.config.add_config(inference_config, "inference")

        pred_info = []

        for img_set_conf in self.config.inference_img_set_confs:
            trial_name = img_set_conf["trial_name"]
            mission_date = img_set_conf["mission_date"] 

            img_set = ImgSet(trial_name, mission_date)
            for dataset_name in img_set_conf["datasets"]:

                patch_dir, is_annotated = self.create_patches(self.config.inference_patch_extraction_params, 
                                                                img_set, dataset_name)
                
                pred_dir = os.path.join(self.config.model_dir, os.path.basename(patch_dir))
                pred_path = os.path.join(pred_dir, "predictions.json")

                if os.path.exists(pred_path):
                    raise RuntimeError("Existing predictions found for model {} (prediction dir: {})".format(
                                        self.config.instance_uuid, pred_dir))
                else:
                    os.makedirs(pred_dir)
                    save_config = copy.deepcopy(inference_config)
                    del save_config["inference_image_sets"]
                    save_config["inference_image_set"] = {
                        "trial_name": trial_name,
                        "mission_date": mission_date,
                        "dataset_name": dataset_name
                    }
                    json_io.save_json(os.path.join(pred_dir, "inference_config.json"), save_config)
                    self.custom_run_inference(patch_dir, pred_dir, img_set.datasets[dataset_name])

                pred_info.append({
                    "trial_name": trial_name,
                    "mission_date": mission_date,
                    "dataset_name": dataset_name,
                    "prediction_dirname": os.path.basename(pred_dir)
                })

        return pred_info



class RetinaNetWrapper(DetectorWrapper):

    def custom_config(self, model_dir):
        return model_config.RetinaNetConfig(model_dir)

    def custom_train_model(self, training_patch_dirs, validation_patch_dirs):
        retinanet_driver.train(training_patch_dirs, validation_patch_dirs, self.config)

    def custom_run_inference(self, patch_dir, pred_dir, img_dataset):
        retinanet_driver.generate_predictions(patch_dir, pred_dir, img_dataset, self.config)


class CenterNetWrapper(DetectorWrapper):

    def custom_config(self, model_dir):
        return model_config.CenterNetConfig(model_dir)

    def custom_train_model(self, training_patch_dirs, validation_patch_dirs):
        centernet_driver.train(training_patch_dirs, validation_patch_dirs, self.config)

    def custom_run_inference(self, patch_dir, pred_dir, img_dataset):
        centernet_driver.generate_predictions(patch_dir, pred_dir, img_dataset, self.config)


class EfficientDetWrapper(DetectorWrapper):

    def custom_config(self, model_dir):
        return model_config.EfficientDetConfig(model_dir)

    def custom_train_model(self, training_patch_dirs, validation_patch_dirs):
        efficientdet_driver.train(training_patch_dirs, validation_patch_dirs, self.config)

    def custom_run_inference(self, patch_dir, pred_dir, img_dataset):
        efficientdet_driver.generate_predictions(patch_dir, pred_dir, img_dataset, self.config)



def create_model_wrapper(model_dir):

    arch_config_path = os.path.join(model_dir, "arch_config.json")

    if not os.path.exists(arch_config_path):
        raise RuntimeError("No architecture configuration file found for '{}'.".format(instance_name))

    arch_config = json_io.load_json(arch_config_path)

    model_type = arch_config["model_type"]

    if model_type == "retinanet":
        model_wrapper = RetinaNetWrapper(model_dir, arch_config)
    elif model_type == "centernet":
        model_wrapper = CenterNetWrapper(model_dir, arch_config)
    elif model_type == "efficientdet":
        model_wrapper = EfficientDetWrapper(model_dir, arch_config)
    else:
        raise RuntimeError("Unknown model type: '{}'.".format(model_type))

    return model_wrapper




def create_model(req_args):

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")

    if "instance_uuid" not in req_args:
        req_args["instance_uuid"] = str(uuid.uuid4())

    if "instance_name" not in req_args:
        req_args["instance_name"] = randomname.get_name()

    instance_uuid = req_args["instance_uuid"]
    model_dir = os.path.join(usr_data_root, "models", instance_uuid)
    weights_dir = os.path.join(model_dir, "weights")
    arch_config_path = os.path.join(model_dir, "arch_config.json")


    if os.path.exists(model_dir):
        raise RuntimeError("Found existing model with uuid '{}'.".format(instance_uuid))

    os.makedirs(model_dir)
    os.makedirs(weights_dir)
    
    json_io.save_json(arch_config_path, req_args)

    logger.info("Instantiated new model: '{}' (uuid: {})".format(req_args["instance_name"], req_args["instance_uuid"]))


def train_model(req_args):

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")
    instance_uuid = req_args["instance_uuid"]
    instance_name = req_args["instance_name"]

    logger.info("Started training model: '{}' (uuid: {})".format(instance_name, instance_uuid))

    model_dir = os.path.join(usr_data_root, "models", instance_uuid)

    train_config = req_args
    train_config_path = os.path.join(model_dir, "training_config.json")
    json_io.save_json(train_config_path, train_config)

    model_wrapper = create_model_wrapper(model_dir)

    model_wrapper.train_model(train_config)

    logger.info("Finished training model: '{}' (uuid: {})".format(instance_name, instance_uuid))
    

def add_entry_to_inference_record(trial_name, mission_date, dataset_name, category, key, value):

    usr_data_root = os.path.join("usr", "data")
    inference_lookup_path = os.path.join(usr_data_root, "records", "inference_lookup.json")
    inference_lookup = json_io.load_json(inference_lookup_path)

    if trial_name not in inference_lookup["inference_runs"]:
        inference_lookup["inference_runs"][trial_name] = {}
    if mission_date not in inference_lookup["inference_runs"][trial_name]:
        inference_lookup["inference_runs"][trial_name][mission_date] = {}
    if dataset_name not in inference_lookup["inference_runs"][trial_name][mission_date]:
        inference_lookup["inference_runs"][trial_name][mission_date][dataset_name] = {"models": {},
                                                                                      "groups": {}}
    if category != "models" and category != "groups":
        raise RuntimeError("Invalid category: '{}'".format(category))

    inference_lookup["inference_runs"][trial_name][mission_date][dataset_name][category][key] = value

    json_io.save_json(inference_lookup_path, inference_lookup)




def run_inference(req_args):

    logger = logging.getLogger(__name__)

    usr_data_root = os.path.join("usr", "data")
    instance_uuid = req_args["instance_uuid"]
    instance_name = req_args["instance_name"]

    logger.info("Started running inference with: '{}' (uuid: {})".format(instance_name, instance_uuid))

    model_dir = os.path.join(usr_data_root, "models", instance_uuid)

    inference_config = req_args

    model_wrapper = create_model_wrapper(model_dir)

    pred_info = model_wrapper.run_inference(inference_config)
    #predictions = json_io.load_json(os.path.join(pred_dir, "predictions.json"))

    #trial_name = req_args["trial_name"]
    #mission_date = req_args["mission_date"]
    #dataset_names = req_args["dataset_names"]

    key = instance_uuid
    value = {"instance_uuid": instance_uuid,
             "instance_name" : instance_name} 
    for p in pred_info:
        pred_dirname = p["prediction_dirname"]
        predictions = json_io.load_json(os.path.join(model_dir, pred_dirname, "predictions.json"))
        
        value["prediction_dirname"] = pred_dirname
        value["metrics"] = predictions["metrics"]        

        add_entry_to_inference_record(p["trial_name"], p["mission_date"], p["dataset_name"], "models", key, value)


    logger.info("Finished running inference with: '{}' (uuid: {})".format(instance_name, instance_uuid))

    return pred_info




# def visualize_models(req_args):
    
#     usr_data_root = os.path.join("usr", "data")
#     vis_name = req_args["visualization_name"]

#     vis_dir = os.path.join(user_data_root, "visualizations", vis_name)
#     if os.path.exists(vis_dir):
#         raise RuntimeError("Visualization with name '{}' already exists.".format(vis_name))

#     os.makedirs(vis_dir)

#     img_set_name = req_args["image_set_name"]
    
#     scenario = {"dataset_name": req_args["dataset_name"],
#                 "parameters":   req_args["patch_extraction_params"],
#                 "is_annotated": req_args["is_annotated"]}

#     img_set = ImgSet(user_data_root, img_set_name)
#     scenario_exists, scenario_id = ep.search_for_scenario(img_set, scenario)

#     if not scenario_exists:
#         raise RuntimeError("Patch extraction scenario specified could not be found.")


#     pred_lst = []
#     patch_pred_lst = []
#     img_pred_lst = []
#     pred_stats_lst = []
#     instance_names = req_args["model_instance_names"]
#     for i, instance_name in enumerate(instance_names):

#         model_dir = os.path.join(user_data_root, "models", instance_name)
#         model_out_dir = os.path.join(vis_dir, instance_name)
#         os.makedirs(model_out_dir)

#         model_vis.loss_plot(model_out_dir, instance_name, model_dir)

#         pred_dir = os.path.join(model_dir, scenario_id)
#         pred_path = os.path.join(pred_dir, "predictions.json")
#         pred = json_io.load_json(pred_path)
#         pred_lst.append(pred)

#         inference_config_path = os.path.join(pred_dir, "inference_config.json")
#         inference_config = json_io.load_json(inference_config_path)
#         img_nms_iou_thresh = inference_config["image_nms_iou_thresh"]

#         if i == 0:
#             patch_sample_size = req_args["patch_sample_size"]
#             img_sample_size = req_args["image_sample_size"]
#             patch_paths = get_sample_patch_paths(pred, patch_sample_size)
#             img_paths = get_sample_image_paths(pred, img_sample_size, 
#                                                img_set.class_map, img_nms_iou_thresh)

#         patch_pred_lst.append(decode_predictions.decode_patch_predictions(pred))
#         img_pred_lst.append(decode_predictions.decode_img_predictions(pred,
#                                                                       img_set.class_map,
#                                                                       img_nms_iou_thresh))

#         #patch_dir = os.path.join(model_out_dir, "patch_predictions")
#         #os.makedirs(patch_dir)
#         #model_vis.output_patch_predictions(patch_dir, pred, img_set.class_map, patch_paths=patch_paths)

#         #img_dir = os.path.join(model_out_dir, "image_predictions")
#         #os.makedirs(img_dir)
#         #model_vis.output_image_predictions(img_dir, pred, img_set.class_map, img_nms_iou_thresh, 
#         #                                   img_paths=img_paths)


#         pred_stats_path = os.path.join(pred_dir, "prediction_stats.json")
#         pred_stats = json_io.load_json(pred_stats_path)
#         pred_stats_lst.append(pred_stats)


#     patch_dir = os.path.join(vis_dir, "patch_predictions")
#     image_dir = os.path.join(vis_dir, "image_predictions")
#     os.makedirs(patch_dir)
#     os.makedirs(image_dir)
#     model_vis.output_patch_predictions(patch_dir, patch_pred_lst, img_set.class_map, patch_paths)
#     #model_vis.output_image_predictions(image_dir, img_pred_lst, img_set.class_map, img_paths)
#     model_vis.image_counts(vis_dir, instance_names, pred_stats_lst)
#     model_vis.time_vs_mAP(vis_dir, instance_names, pred_lst, pred_stats_lst)







# def get_sample_patch_paths(predictions, sample_size, must_contain_box=True):

#     is_annotated = predictions["is_annotated"]

#     if must_contain_box and not is_annotated:
#         raise RuntimeError("Incompatible argument 'must_contain_box' == True with unannotated patches.")


#     patch_predictions = decode_predictions.decode_patch_predictions(predictions)
    
#     pool = []
#     if must_contain_box:
#         for patch_path, patch_pred in patch_predictions.items():
#             if np.size(patch_pred["patch_abs_boxes"]) > 0:
#                 pool.append(patch_path)

#     else:
#         pool = patch_predictions.keys()

#     sample_patch_paths = random.sample(pool, min(sample_size, len(pool)))

#     return sample_patch_paths



# def get_sample_image_paths(predictions, sample_size, class_map, img_nms_iou_thresh):

#     is_annotated = predictions["is_annotated"]

#     img_predictions = decode_predictions.decode_img_predictions(predictions, class_map, img_nms_iou_thresh)

#     sample_img_paths = random.sample(img_predictions.keys(), 
#                                      min(sample_size, len(img_predictions.keys())))

#     return sample_img_paths