import logging
import os
import shutil
import tqdm
import imagesize
import numpy as np
import cv2

from io_utils import json_io, xml_io
from models.common import box_utils






class ImgSet(object):

    def __init__(self, farm_name, field_name, mission_date):

        usr_data_root = os.path.join("usr", "data")

        self.farm_name = farm_name
        self.field_name = field_name
        self.mission_date = mission_date
        self.root_dir = os.path.join(usr_data_root, "image_sets", farm_name, field_name, mission_date)
        self.img_dir = os.path.join(self.root_dir, "images")
        self.patch_dir = os.path.join(self.root_dir, "patches")
        self.dzi_dir = os.path.join(self.root_dir, "dzi_images")

        img_set_config_path = os.path.join(self.root_dir, "image_set_config.json")
        img_set_info = json_io.load_json(img_set_config_path)
        
        self.class_map = img_set_info["class_map"]
        self.reverse_class_map = {v: k for k, v in self.class_map.items()}
        self.num_classes = img_set_info["num_classes"]
        
        self.datasets = {
            "training": DataSet(img_set_info["training_image_paths"], self.class_map, "training"),
            "validation": DataSet(img_set_info["validation_image_paths"], self.class_map, "validation"),
            "test": DataSet(img_set_info["test_image_paths"], self.class_map, "test"),
            "all": DataSet(img_set_info["all_image_paths"], self.class_map, "all")

        }

        if (not self.datasets["training"].is_annotated) or \
           (not self.datasets["validation"].is_annotated) or \
           (not self.datasets["test"].is_annotated):

           raise RuntimeError("Training, validation, and test datasets must only contain annotated images.")


        #img_width, img_height = imagesize.get(img_set_info["training_image_paths"][0])
        #self.img_width = img_width
        #self.img_height = img_height
        #self.flight_metadata = img_set_info["flight_metadata"]

    def get_box_counts(self):

        box_counts = {}
        for dataset_name, dataset in self.datasets.items():
            box_counts[dataset_name] = dataset.get_box_counts(self.class_map)
        return box_counts

class DataSet(object):

    def __init__(self, img_paths, class_map, name):

        self.imgs = []
        self.class_map = class_map

        self.is_annotated = True
        for img_path in img_paths:
            img = Img(img_path)
            if not img.is_annotated:
                self.is_annotated = False
            self.imgs.append(img)


        self.name = name

    def get_box_counts(self):
        xml_paths = [img.xml_path for img in self.imgs if img.is_annotated]
        box_counts = xml_io.get_box_counts(xml_paths, self.class_map)
        return box_counts


    def get_mean_box_area(self):
        box_areas = []

        for img in self.imgs:
            boxes, _ = xml_io.load_boxes_and_classes(img.xml_path, self.class_map)
            box_areas.extend(box_utils.box_areas_np(boxes).tolist())

        return np.mean(box_areas)

class Img(object):

    def __init__(self, img_path):

        self.img_name = os.path.basename(img_path)[:-4]
        self.img_path = img_path
        self.xml_path = img_path[:-3] + "xml"
        self.is_annotated = os.path.exists(self.xml_path)


    def load_img_array(self):
        return cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)

    def get_wh(self):
        w, h = imagesize.get(self.img_path)
        return w, h

    def get_gsd(self, flight_metadata=None, prioritize_exif=False):
        metadata = exif_io.get_exif_metadata(self.img_path)

        # TODO
        #sensor_x_res = metadata["EXIF:FocalPlaneXResolution"]
        #sensor_y_res = metadata["EXIF:FocalPlaneYResolution"]
        #focal_length = metadata["EXIF:FocalLength"]

        flight_height = flight_metadata["flight_height_m"]
        sensor_height = flight_metadata["sensor_height_mm"] * 1000
        sensor_width = flight_metadata["sensor_width_mm"] * 1000
        focal_length = flight_metadata["focal_length_mm"] * 1000

        img_width, img_height = self.get_wh()

        gsd_h = (flight_height * sensor_height) / (focal_length * img_height)
        gsd_w = (flight_height * sensor_width) / (focal_length * img_width)
        gsd = max(gsd_h, gsd_w)
        
        return gsd

    def get_area_m2(self, flight_metadata=None, prioritize_exif=False):

        gsd = self.get_gsd(flight_metadata=flight_metadata, prioritize_exif=prioritize_exif)
        img_width, img_height = self.get_wh()
        area_m2 = (img_width * gsd) * (img_height * gsd)
        return area_m2


def register_image_set(req_args):

    logger = logging.getLogger(__name__)

    farm_name = req_args["farm_name"]
    field_name = req_args["field_name"]
    mission_date = req_args["mission_date"]
    usr_data_root = os.path.join("usr", "data")


    usr_models_dir = os.path.join(usr_data_root, "models")
    if not os.path.exists(usr_models_dir):
        os.makedirs(usr_models_dir)
    usr_groups_dir = os.path.join(usr_data_root, "groups")
    if not os.path.exists(usr_groups_dir):
        os.makedirs(usr_groups_dir)
    usr_records_dir = os.path.join(usr_data_root, "records")
    if not os.path.exists(usr_records_dir):
        os.makedirs(usr_records_dir)
    usr_ensembles_dir = os.path.join(usr_data_root, "ensembles")
    if not os.path.exists(usr_ensembles_dir):
        os.makedirs(usr_ensembles_dir)
    inference_lookup_path = os.path.join(usr_records_dir, "inference_lookup.json")
    if not os.path.exists(inference_lookup_path):
        json_io.save_json(inference_lookup_path, {"inference_runs": {}})



    img_set_dir = os.path.join(usr_data_root, "image_sets", farm_name, field_name, mission_date)

    img_set_img_dir = os.path.join(img_set_dir, "images")
    img_set_patch_dir = os.path.join(img_set_dir, "patches")
    img_set_dzi_dir = os.path.join(img_set_dir, "dzi_images")

    img_set_config_path = os.path.join(img_set_dir, "image_set_config.json")

    if not os.path.exists(img_set_img_dir):
        raise RuntimeError("Directory containing images for the image set does not exist: {}".format(img_set_img_dir))
    if os.path.exists(img_set_patch_dir):
        logger.info("Removing existing directory for storing image patches: {}.".format(img_set_patch_dir))
        shutil.rmtree(img_set_patch_dir)
    if os.path.exists(img_set_dzi_dir):
        logger.info("Removing existing directory for storing DZI images: {}".format(img_set_dzi_dir))
        shutil.rmtree(img_set_dzi_dir)
    if os.path.exists(img_set_config_path):
        logger.info("Removing existing image set configuration file: {}".format(img_set_config_path))
        os.remove(img_set_config_path)



    os.makedirs(img_set_patch_dir)
    os.makedirs(img_set_dzi_dir)
    patch_scenario_lookup_path = os.path.join(img_set_patch_dir, "scenario_lookup.json")
    patch_scenario_lookup = {"scenarios": {}}
    json_io.save_json(patch_scenario_lookup_path, patch_scenario_lookup)

    img_set_config = {}

    img_set_config["all_image_paths"] = [os.path.join(img_set_img_dir, name) for name in os.listdir(img_set_img_dir) if 
                                   name.endswith("JPG") or name.endswith("jpg") or name.endswith("tif")]

    img_set_config["all_annotation_paths"] = [os.path.join(img_set_img_dir, name) for name in os.listdir(img_set_img_dir) if 
                                          name.endswith("XML") or name.endswith("xml")]

    img_set_config["num_images"] = len(img_set_config["all_image_paths"])
    img_set_config["num_annotated_images"] = len(img_set_config["all_annotation_paths"])


    if "training_images" in req_args and \
       "validation_images" in req_args and \
       "test_images" in req_args:

       img_set_config["training_image_paths"] = [os.path.join(img_set_img_dir, name) for name in req_args["training_images"]]
       img_set_config["validation_image_paths"] = [os.path.join(img_set_img_dir, name) for name in req_args["validation_images"]]
       img_set_config["test_image_paths"] = [os.path.join(img_set_img_dir, name) for name in req_args["test_images"]]

    elif "training_pct" in req_args and \
         "validation_pct" in req_args and \
         "test_pct" in req_args:

        num_training = m.floor(img_set_config["num_annotations"] * req_args["training_pct"])
        num_validation = m.floor(img_set_config["num_annotations"] * req_args["validation_pct"])

        shuffled = random.sample(img_set_config["all_annotation_paths"], img_set_config["num_annotations"])

        img_set_config["training_image_paths"] = shuffled[:num_training]
        img_set_config["validation_image_paths"] = shuffled[num_training:(num_training+num_validation)]
        img_set_config["test_image_paths"] = shuffled[(num_training+num_validation):]

    else:
        raise RuntimeError("Missing information needed to create dataset splits.")


    img_set_config["num_training_images"] = len(img_set_config["training_image_paths"])
    img_set_config["num_validation_images"] = len(img_set_config["validation_image_paths"])
    img_set_config["num_test_images"] = len(img_set_config["test_image_paths"])

    img_set_config["class_map"] = xml_io.create_class_map(img_set_config["all_annotation_paths"])
    img_set_config["num_classes"] = len(img_set_config["class_map"].keys())



    all_img_paths = []
    all_img_paths.extend(img_set_config["training_image_paths"])
    all_img_paths.extend(img_set_config["validation_image_paths"]) 
    all_img_paths.extend(img_set_config["test_image_paths"])
    #all_img_paths = img_set_config["all_image_paths"]

    conversion_tmp_dir = os.path.join(img_set_dzi_dir, "conversion_tmp")
    os.mkdir(conversion_tmp_dir)

    for img_path in tqdm.tqdm(all_img_paths, desc="Generating DZI images"):

        img_dzi_path = os.path.join(img_set_dzi_dir, os.path.basename(img_path)[:-4] + ".dzi")

        if img_path[-4] != ".jpg" and img_path[-4] != ".png":
            tmp_path = os.path.join(conversion_tmp_dir, os.path.basename(img_path)[:-4] + ".jpg")
            os.system("convert " + img_path + " " + tmp_path)
            os.system("./MagickSlicer/magick-slicer.sh " + tmp_path + " " + img_dzi_path[:-4])
            os.remove(tmp_path)
        else:
            os.system("./MagickSlicer/magick-slicer.sh " + img_path + " " + img_dzi_path[:-4])

    os.rmdir(conversion_tmp_dir)

    json_io.save_json(img_set_config_path, img_set_config)

    img_set = ImgSet(farm_name, field_name, mission_date)
    box_counts = img_set.get_box_counts()
    img_set_config["box_counts"] = box_counts
    json_io.save_json(img_set_config_path, img_set_config)