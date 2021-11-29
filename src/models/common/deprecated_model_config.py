from abc import ABC, abstractmethod
import os
import shutil
import logging

from image_set import ImgSet
from io_utils import json_io
from models.common import class_map_utils


class ModelConfig(ABC):


    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.weights_dir = os.path.join(model_dir, "weights")



    def add_config(self, config, config_type):
        required_keys = self.required_keys(config_type)
        self.check_config(config, required_keys)
        self.process_config(config, config_type)


    def required_keys(self, config_type):
        if config_type == "arch":
            keys = ["model_uuid",
                    "model_name",
                    "model_type"]

        elif config_type == "training":
            keys = ["model_uuid",
                    "model_name",
                    "training_image_sets",
                    "training_patch_extraction_params",
                    "validation_patch_extraction_params",
                    "learning_rate",
                    "num_epochs",
                    "early_stopping",
                    "batch_size",
                    "data_augmentations",
                    "percent_of_training_set_used",
                    "save_method"]

        elif config_type == "inference":
            keys = ["model_uuid",
                    "model_name",
                    "inference_image_sets",
                    "inference_patch_extraction_params",
                    "batch_size",
                    "image_nms_iou_thresh",
                    "score_thresh"]
        else:
            raise RuntimeError("Unknown configuration type.")

        keys = keys + self.custom_required_keys(config_type)
        return keys


    @abstractmethod
    def custom_required_keys(self, config_type):
        pass





    def process_config(self, config, config_type):

        if config_type == "arch":
            self.model_type = config["model_type"]
            self.model_uuid = config["model_uuid"]
            self.model_name = config["model_name"]

        elif config_type == "training":
            self.training_img_set_confs = []
            for training_img_set_conf in config["training_image_sets"]:
                self.training_img_set_confs.append(training_img_set_conf)
            class_map_utils.create_and_save_class_map_data(config)
            class_map_data = class_map_utils.load_class_map_data(config["model_uuid"])
            self.class_map = class_map_data["class_map"]
            self.reverse_class_map = class_map_data["reverse_class_map"]
            self.num_classes = class_map_data["num_classes"]
            self.training_patch_extraction_params = config["training_patch_extraction_params"]
            self.validation_patch_extraction_params = config["validation_patch_extraction_params"]
            self.learning_rate = config["learning_rate"]
            self.num_epochs = config["num_epochs"]
            self.early_stopping = config["early_stopping"]
            self.batch_size = config["batch_size"]
            self.data_augmentations = config["data_augmentations"]
            self.percent_of_training_set_used = config["percent_of_training_set_used"]
            self.save_method = config["save_method"]


        elif config_type == "inference":
            self.inference_img_set_confs = []
            for inference_image_set_conf in config["inference_image_sets"]: 
                self.inference_img_set_confs.append(inference_image_set_conf)
            class_map_data = class_map_utils.load_class_map_data(config["model_uuid"])
            self.class_map = class_map_data["class_map"]
            self.reverse_class_map = class_map_data["reverse_class_map"]
            self.num_classes = class_map_data["num_classes"]
            self.inference_patch_extraction_params = config["inference_patch_extraction_params"]
            self.batch_size = config["batch_size"]
            self.img_nms_iou_thresh = config["image_nms_iou_thresh"]
            self.score_thresh = config["score_thresh"]


        self.custom_process_config(config, config_type)

    @abstractmethod
    def custom_process_config(self):
        pass


    def check_config(self, config, required_keys):

        for required_key in required_keys:
            if required_key not in config:
                raise RuntimeError("Missing required configuration parameter: {}.".format(required_key))





class RetinaNetConfig(ModelConfig):

    def custom_required_keys(self, config_type):
        if config_type == "arch":
            keys = ["backbone_config",
                    "max_detections",
                    "max_detections_per_class",
                    "input_img_min_side",
                    "input_img_max_side",
                    "patch_nms_iou_thresh"]
        elif config_type == "training":
            keys = []
        elif config_type == "inference":
            keys = []
        else:
            raise RuntimeError("Unknown config type: {}".format(config_type))

        return keys

    def custom_process_config(self, config, config_type):

        if config_type == "arch":
            self.backbone_config = config["backbone_config"]
            self.max_detections = config["max_detections"]
            self.max_detections_per_class = config["max_detections_per_class"]
            self.input_img_min_side = config["input_img_min_side"]
            self.input_img_max_side = config["input_img_max_side"]
            self.patch_nms_iou_thresh = config["patch_nms_iou_thresh"]

            self.gamma = 2.0
            self.alpha = 0.25
            self.delta = 1.0

        elif config_type == "training":
            pass
        elif config_type == "inference":
            pass
        else:
            raise RuntimeError("Unknown config type: {}".format(config_type))



class CenterNetConfig(ModelConfig):

    def custom_required_keys(self, config_type):
        if config_type == "arch":
            keys = ["backbone_config",
                    "max_detections"]
        elif config_type == "training":
            keys = []
        elif config_type == "inference":
            keys = []
        else:
            raise RuntimeError("Unknown config type: {}".format(config_type))

        return keys


    def custom_process_config(self, config, config_type):


        if config_type == "arch":
            input_img_shape = {
                "resnet18": (384, 384),
                "resnet34": (384, 384),
                "resnet50": (384, 384),
                "resnet101": (384, 384), 
                "resnet152": (384, 384),

                "D0": (512, 512), "D1": (640, 640), "D2": (768, 768),
                "D3": (896, 896), "D4": (1024, 1024), "D5": (1280, 1280),
                "D6": (1408, 1408), "D7": (1536, 1536)
            }

            downsampling_ratio = {
                "resnet18": 4, "resnet34": 4, "resnet50": 4, "resnet101": 4, "resnet152": 4,
                "D0": 8, "D1": 8, "D2": 8, "D3": 8, "D4": 8, "D5": 8, "D6": 8, "D7": 8
            }

            # efficientdet
            #width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
            #depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
            #dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}
            
            # bifpn channels
            bifpn_width = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
            
            # bifpn layers
            bifpn_depth = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}

            head_conv = {
                        "no_conv_layer": 0, 
                        "resnet18": 64, "resnet34": 64, "resnet50": 64, "resnet101": 64, "resnet152": 64,
                        "D0": bifpn_width["D0"], "D1": bifpn_width["D1"], "D2": bifpn_width["D2"], "D3": bifpn_width["D3"],
                        "D4": bifpn_width["D4"], "D5": bifpn_width["D5"], "D6": bifpn_width["D6"], "D7": bifpn_width["D7"]
            }

            self.backbone_config = config["backbone_config"]
            self.max_detections = config["max_detections"]


            self.input_img_shape = input_img_shape[self.backbone_config["backbone_type"]]
            self.downsampling_ratio = downsampling_ratio[self.backbone_config["backbone_type"]]

            self.head_conv = head_conv[self.backbone_config["backbone_type"]]

            self.bifpn_width = bifpn_width[self.backbone_config["backbone_type"]] if self.backbone_config["backbone_type"] in bifpn_width else None
            self.bifpn_depth = bifpn_depth[self.backbone_config["backbone_type"]] if self.backbone_config["backbone_type"] in bifpn_depth else None
            #self.dropout_rate = dropout_rate[self.backbone_type] if self.backbone_type in dropout_rate else None
            #self.width_coefficient = width_coefficient[self.backbone_type] if self.backbone_type in width_coefficient else None
            #self.depth_coefficient = depth_coefficient[self.backbone_type] if self.backbone_type in depth_coefficient else None

            # weighting of loss components
            self.heatmap_loss_weight = 1.0
            self.size_loss_weight = 0.1
            self.offset_loss_weight = 1.0

        elif config_type == "training":
            pass
        elif config_type == "inference":
            pass
        else:
            raise RuntimeError("Unknown config type: {}".format(config_type))



class EfficientDetConfig(ModelConfig):

    def custom_required_keys(self, config_type):
        if config_type == "arch":
            keys = ["backbone_config",
                    "max_detections",
                    "max_detections_per_class",
                    "patch_nms_iou_thresh"]

        elif config_type == "training":
            keys = []
        elif config_type == "inference":
            keys = []
        else:
            raise RuntimeError("Unknown config type: {}".format(config_type))

        return keys

    def custom_process_config(self, config, config_type):

        if config_type == "arch":



            self.backbone_config = config["backbone_config"]
            self.max_detections = config["max_detections"]
            self.max_detections_per_class = config["max_detections_per_class"]
            self.patch_nms_iou_thresh = config["patch_nms_iou_thresh"]


            self.gamma = 2.0    # original paper uses gamma=1.5, alpha=0.25
            self.alpha = 0.25
            self.delta = 1.0

            input_img_shape = {
                "D0": (512, 512), "D1": (640, 640), "D2": (768, 768),
                "D3": (896, 896), "D4": (1024, 1024), "D5": (1280, 1280),
                "D6": (1408, 1408), "D7": (1536, 1536)
            }


            # bifpn channels
            bifpn_width = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
            
            # bifpn layers
            bifpn_depth = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}

            #head_conv = {
            #            "no_conv_layer": 0, 
            #            "D0": bifpn_width["D0"], "D1": bifpn_width["D1"], "D2": bifpn_width["D2"], "D3": bifpn_width["D3"],
            #            "D4": bifpn_width["D4"], "D5": bifpn_width["D5"], "D6": bifpn_width["D6"], "D7": bifpn_width["D7"]
            #}

            head_layers = {"D0": 3, "D1": 3, "D2": 3, "D3": 4, "D4": 4, "D5": 4, "D6": 5, "D7": 5}

            self.input_img_shape = input_img_shape[self.backbone_config["backbone_type"]]

            #self.head_conv = head_conv[self.backbone_type]

            self.bifpn_width = bifpn_width[self.backbone_config["backbone_type"]]
            self.bifpn_depth = bifpn_depth[self.backbone_config["backbone_type"]]
            self.head_layers = head_layers[self.backbone_config["backbone_type"]]


        elif config_type == "training":
            pass
        elif config_type == "inference":
            pass
        else:
            raise RuntimeError("Unknown config type: {}".format(config_type))
