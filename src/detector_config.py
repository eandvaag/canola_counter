from abc import ABC, abstractmethod
import randomname
import os
import shutil
import logging


from io_utils import json_io


class DetectorConfig(ABC):


    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def required_config_params(self):
        pass

    @abstractmethod
    def process_config(self, config):
        pass


    def __init__(self, settings, instance_name):

        logger = logging.getLogger(__name__)

        new_instance = instance_name is None

        if new_instance:
            self.instance_name = randomname.get_name()
        else:
            self.instance_name = instance_name

        self.model_dir = os.path.join(settings.detectors_dir, self.instance_name)
        self.weights_dir = os.path.join(self.model_dir, "weights")
        self.config_path = os.path.join(self.model_dir, "model_config.json")

        if new_instance:
            os.makedirs(self.model_dir)
            os.makedirs(self.weights_dir)
            shutil.copy(settings.detector_config_path, self.config_path)

        self.model_name = self.model_name()

        self.load_config()


        status = "new" if new_instance else "existing"
        logger.info("\n\nInitialized {} {} model: '{}'.\n\n".format(status, self.model_name, self.instance_name))


    def load_config(self):
        config = json_io.load_json(self.config_path)

        req_params = self.required_config_params()
        for req_param in req_params:
            if req_param not in config:
                raise RuntimeError("Missing model configuration parameter for {}: {}.".format(self.model_name, req_param))

        for param in config:
            if param not in req_params:
                raise RuntimeError("Unrecognized model configuration parameter for {}: {}".format(self.model_name, param))


        self.process_config(config)



class RetinaNetConfig(DetectorConfig):

    def model_name(self):
        return "RetinaNet"

    def required_config_params(self):
        return ["num_classes",
                "score_thresh",
                "backbone_name",
                "nms_iou_thresh",
                "max_detections",
                "max_detections_per_class",
                "learning_rate",
                "num_epochs",
                "early_stopping",
                "train_batch_size",
                "inference_batch_size",
                "input_img_min_side",
                "input_img_max_side",
                "smallest_fmap_stride",
                "input_img_channels",
                "load_method",
                "data_augmentations"]

    def process_config(self, config):

        self.num_classes = config["num_classes"]
        self.score_thresh = config["score_thresh"]
        self.backbone_name = config["backbone_name"]
        self.nms_iou_thresh = config["nms_iou_thresh"]
        self.max_detections = config["max_detections"]
        self.max_detections_per_class = config["max_detections_per_class"]
        self.learning_rate = config["learning_rate"]
        self.num_epochs = config["num_epochs"]
        self.early_stopping = config["early_stopping"]
        self.train_batch_size = config["train_batch_size"]
        self.inference_batch_size = config["inference_batch_size"]
        self.input_img_min_side = config["input_img_min_side"]
        self.input_img_max_side = config["input_img_max_side"]
        self.smallest_fmap_stride = config["smallest_fmap_stride"]
        self.input_img_channels = config["input_img_channels"]
        self.load_method = config["load_method"]
        self.data_augmentations = config["data_augmentations"]


class CenterNetConfig(DetectorConfig):


    def model_name(self):
        return "CenterNet"

    def required_config_params(self):
        return ["num_classes",
                "score_thresh",
                "backbone_name",
                "max_detections",
                "learning_rate",
                "num_epochs",
                "early_stopping",
                "train_batch_size",
                "inference_batch_size",
                "load_method",
                "data_augmentations"]


    def process_config(self, config):

        input_img_shape = {
            "resnet_18": (384, 384), "resnet_34": (384, 384), "resnet_50": (384, 384),
            "resnet_101": (384, 384), "resnet_152": (384, 384),

            "prebuilt_resnet_50": (384, 384),

            "D0": (512, 512), "D1": (640, 640), "D2": (768, 768),
            "D3": (896, 896), "D4": (1024, 1024), "D5": (1280, 1280),
            "D6": (1408, 1408), "D7": (1536, 1536)
        }

        downsampling_ratio = {
            "resnet_18": 4, "resnet_34": 4, "resnet_50": 4, "resnet_101": 4, "resnet_152": 4,
            "prebuilt_resnet_50": 4,
            "D0": 8, "D1": 8, "D2": 8, "D3": 8, "D4": 8, "D5": 8, "D6": 8, "D7": 8
        }

        # efficientdet
        width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
        depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
        dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}
        # bifpn channels
        w_bifpn = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
        # bifpn layers
        d_bifpn = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}

        head_conv = {
                    "no_conv_layer": 0, 
                    "resnet_18": 64, "resnet_34": 64, "resnet_50": 64, "resnet_101": 64, "resnet_152": 64,
                    "prebuilt_resnet_50": 64,
                    "dla": 256,
                    "D0": w_bifpn["D0"], "D1": w_bifpn["D1"], "D2": w_bifpn["D2"], "D3": w_bifpn["D3"],
                    "D4": w_bifpn["D4"], "D5": w_bifpn["D5"], "D6": w_bifpn["D6"], "D7": w_bifpn["D7"]
        }

        config = json_io.load_json(self.config_path)

        self.num_classes = config["num_classes"]
        self.score_thresh = config["score_thresh"]
        self.backbone_name = config["backbone_name"]
        self.max_detections = config["max_detections"]
        self.learning_rate = config["learning_rate"]
        self.num_epochs = config["num_epochs"]
        self.early_stopping = config["early_stopping"]
        self.train_batch_size = config["train_batch_size"]
        self.inference_batch_size = config["inference_batch_size"]
        self.load_method = config["load_method"]
        self.data_augmentations = config["data_augmentations"]




        self.input_img_shape = input_img_shape[self.backbone_name]
        self.downsampling_ratio = downsampling_ratio[self.backbone_name]
        #self.heads = { "heatmap": self.num_classes, "wh": 2, "reg": 2 }

        self.head_conv = head_conv[self.backbone_name]
        self.dropout_rate = dropout_rate[self.backbone_name] if self.backbone_name in dropout_rate else None
        self.width_coefficient = width_coefficient[self.backbone_name] if self.backbone_name in width_coefficient else None
        self.depth_coefficient = depth_coefficient[self.backbone_name] if self.backbone_name in depth_coefficient else None

        # weighting of loss components
        self.heatmap_loss_weight = 1.0
        self.size_loss_weight = 0.1
        self.offset_loss_weight = 1.0
