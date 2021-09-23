import os
from pathlib import Path
import math as m
import random

from io_utils import json_io

class Settings:

	def __init__(self, config):

		self.pct_train = config["data_split"]["train"]
		self.pct_val = config["data_split"]["val"]
		self.pct_test = config["data_split"]["test"]

		self.class_map = config["class_map"]


		self.workspace_dir = config["workspace_dir"]

		self.images_dir = os.path.join(self.workspace_dir, "images")
		self.datasets_record_path = os.path.join(self.images_dir, "datasets_record.json")

		self.patches_dir = os.path.join(self.workspace_dir, "patches")
		self.detector_patches_dir = os.path.join(self.patches_dir, "detector_patches")
		self.classifier_patches_dir = os.path.join(self.patches_dir, "classifier_patches")
		#self.detector_scenario_lookup_path = os.path.join(self.detector_patches_dir, "scenario_lookup.json")
		#if not os.path.exists(self.detector_scenario_lookup_path):
		#	json_io.save_json(self.detector_scenario_lookup_path, {})

		self.models_dir = os.path.join(self.workspace_dir, "models")
		self.detectors_dir = os.path.join(self.models_dir, "detectors")
		self.classifiers_dir = os.path.join(self.models_dir, "classifiers")

		self.registration_dir = os.path.join(self.workspace_dir, "registration")



		self.detector_config_path = config["detector_config_path"]
		self.classifier_config_path = config["classifier_config_path"]


		self.detector_train_extraction_method = config["detector_extraction_params"]["train_extraction_method"]
		self.detector_test_extraction_method = config["detector_extraction_params"]["test_extraction_method"]
		self.detector_patch_size = config["detector_extraction_params"]["patch_size"]
		self.detector_tile_patch_overlap = config["detector_extraction_params"]["tile_patch_overlap"]

		self.classifier_train_extraction_method = config["classifier_extraction_params"]["train_extraction_method"]
		self.classifier_test_extraction_method = config["classifier_extraction_params"]["test_extraction_method"]
		self.classifier_patch_padding = config["classifier_extraction_params"]["patch_padding"]


		self._create_workspace()
		self._get_image_paths()
		self._create_datasets()


	def _create_workspace(self):

		dirs_to_check = [self.workspace_dir,
						 self.images_dir]

		for directory in dirs_to_check:
			if not os.path.exists(directory):
				raise RuntimeError("Missing required directory: {}".format(directory))


		dirs_to_create = [self.patches_dir, 
						  self.detector_patches_dir, 
						  self.classifier_patches_dir,
						  self.models_dir,
						  self.registration_dir]

		for directory in dirs_to_create:
			if not os.path.exists(directory):
				os.makedirs(directory)

		return


	def _get_image_paths(self):

		self.img_paths = sorted([os.path.join(self.images_dir, name) for name in os.listdir(self.images_dir) if 
								 name.endswith("JPG") or name.endswith("jpg") or name.endswith("tif")])

		self.annotated_img_paths = [img_path for img_path in self.img_paths if os.path.exists(img_path[:-3] + "xml")]

		self.unannotated_img_paths = [img_path for img_path in self.img_paths if not os.path.exists(img_path[:-3] + "xml")]



	def _create_datasets(self):

		if os.path.exists(self.datasets_record_path):
			self.datasets_record = json_io.load_json(self.datasets_record_path)
			
			self.train_img_paths = self.datasets_record["train_img_paths"]
			self.val_img_paths = self.datasets_record["val_img_paths"]
			self.test_img_paths  = self.datasets_record["test_img_paths"]

		else:

			num_annotated = len(self.annotated_img_paths)
			num_train = m.floor(self.pct_train * num_annotated)
			num_val = m.floor(self.pct_val * num_annotated)
			num_test = num_annotated - (num_train + num_val)


			shuffled = random.sample(self.annotated_img_paths, len(self.annotated_img_paths))

			self.train_img_paths = shuffled[:num_train]
			self.val_img_paths = shuffled[num_train:(num_train+num_val)]
			self.test_img_paths = shuffled[(num_train+num_val):]

			self.datasets_record = {
				"train_img_paths": self.train_img_paths,
				"val_img_paths": self.val_img_paths,
				"test_img_paths": self.test_img_paths
			}

			json_io.save_json(self.datasets_record_path, self.datasets_record)
