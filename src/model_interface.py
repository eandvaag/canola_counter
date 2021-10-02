from abc import ABC, abstractmethod

from detector_config import RetinaNetConfig, CenterNetConfig

from models.detectors.retinanet import retinanet_driver
from models.detectors.centernet import centernet_driver

import model_vis

class DetectorModel(ABC):

    @abstractmethod
    def train(self, train_patch_dir, val_patch_dir):
        pass

    @abstractmethod
    def generate_predictions(self, patch_dir, skip_if_found=True):
        pass


class CenterNet(DetectorModel):

    def __init__(self, settings, instance_name=None):
        super().__init__()
        self.config = CenterNetConfig(settings, instance_name=instance_name)

    def train(self, train_patch_dir, val_patch_dir):
        centernet_driver.train(train_patch_dir, val_patch_dir, self.config)

    def generate_predictions(self, patch_dir, skip_if_found=True):
        return centernet_driver.generate_predictions(patch_dir, self.config, skip_if_found=skip_if_found)


class RetinaNet(DetectorModel):

    def __init__(self, settings, instance_name=None):
        super().__init__()
        self.config = RetinaNetConfig(settings, instance_name=instance_name)

    def train(self, train_patch_dir, val_patch_dir):
        retinanet_driver.train(train_patch_dir, val_patch_dir, self.config)

    def generate_predictions(self, patch_dir, skip_if_found=True):
        return retinanet_driver.generate_predictions(patch_dir, self.config, skip_if_found=skip_if_found)