from abc import ABC
import uuid
import os
import shutil


from io_utils import json_io

from detector_models.retinanet import retinanet

import model_vis

class DetectorModel(ABC):


    def __init__(self, settings, model_uuid=None):

        super().__init__()
        
        if model_uuid is None:
            self.model_uuid = str(uuid.uuid4())
            self.model_dir = os.path.join(settings.detectors_dir, self.model_uuid)
            #self.config = json_io.load_json(settings.detector_config_path)
            os.makedirs(self.model_dir)
            os.makedirs(os.path.join(self.model_dir, "weights"))
            #json_io.save_json(os.path.join(self.model_dir, "model_config.json"), self.config)
            shutil.copy(settings.detector_config_path, os.path.join(self.model_dir, "model_config.json"))
            #json_io.save_json(os.path.join(self.model_dir, "run_settings.json"), settings)
            #self.load_model = False
        else:
            self.model_uuid = model_uuid
            self.model_dir = os.path.join(settings.detectors_dir, self.model_uuid)
            #self.config = json_io.load_json(os.path.join(self.model_dir, "model_config.json"))
            #self.load_model = True

    @property
    def name(self):
        pass


class CenterNet(DetectorModel):

    @property
    def name(self):
        return "CenterNet"
        

class RetinaNet(DetectorModel):

    @property
    def name(self):
        return "RetinaNet"

    def train(self, train_patches_dir, val_patches_dir):
        retinanet.train(train_patches_dir, val_patches_dir, self.model_dir)

    def generate_predictions(self, tf_record_path, found_behaviour="skip"):
        return retinanet.generate_predictions(tf_record_path, self.model_dir, found_behaviour=found_behaviour)