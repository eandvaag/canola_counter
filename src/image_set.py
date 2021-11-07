import os
import shutil
import cv2
import tqdm
import imagesize
import logging

from io_utils import json_io, xml_io






class ImgSet(object):

    def __init__(self, trial_name, mission_date):

        usr_data_root = os.path.join("usr", "data")

        self.trial_name = trial_name
        self.mission_date = mission_date
        self.root_dir = os.path.join(usr_data_root, "image_sets", trial_name, mission_date)
        self.img_dir = os.path.join(self.root_dir, "images")
        self.patch_dir = os.path.join(self.root_dir, "patches")
        self.dzi_dir = os.path.join(self.root_dir, "dzi_images")

        img_set_config_path = os.path.join(self.root_dir, "image_set_config.json")
        img_set_info = json_io.load_json(img_set_config_path)

        self.datasets = {
            "training": DataSet(img_set_info["training_image_paths"], "training"),
            "validation": DataSet(img_set_info["validation_image_paths"], "validation"),
            "test": DataSet(img_set_info["test_image_paths"], "test"),
            "all": DataSet(img_set_info["all_image_paths"], "all")

        }

        if (not self.datasets["training"].is_annotated) or \
           (not self.datasets["validation"].is_annotated) or \
           (not self.datasets["test"].is_annotated):

           raise RuntimeError("Training, validation, and test datasets must only contain annotated images.")

        self.class_map = img_set_info["class_map"]
        self.num_classes = img_set_info["num_classes"]
        img_width, img_height = imagesize.get(img_set_info["training_image_paths"][0])
        self.img_width = img_width
        self.img_height = img_height

class DataSet(object):

    def __init__(self, img_paths, name):

        self.imgs = []
        self.is_annotated = True

        for img_path in img_paths:
            img = Img(img_path)
            if not img.is_annotated:
                self.is_annotated = False
            self.imgs.append(img)


        self.name = name



class Img(object):

    def __init__(self, img_path):

        self.img_name = os.path.basename(img_path)[:-4]
        self.img_path = img_path
        self.xml_path = img_path[:-3] + "xml"
        self.is_annotated = os.path.exists(self.xml_path)


    def load_img_array(self):
        return cv2.imread(self.img_path)








def register_image_set(req_args):

    logger = logging.getLogger(__name__)

    trial_name = req_args["trial_name"]
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
    group_lookup_path = os.path.join(usr_records_dir, "group_lookup.json")
    if not os.path.exists(group_lookup_path):
        json_io.save_json(group_lookup_path, { "groups": {}})
    inference_lookup_path = os.path.join(usr_records_dir, "inference_lookup.json")
    if not os.path.exists(inference_lookup_path):
        json_io.save_json(inference_lookup_path, {"inference_runs": {}})



    img_set_dir = os.path.join(usr_data_root, "image_sets", trial_name, mission_date)

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
    img_set_config["num_annotations"] = len(img_set_config["all_annotation_paths"])


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
        num_test = m.floor(img_set_config["num_annotations"] * req_args["test_pct"])

        shuffled = random.sample(img_set_config["all_annotation_paths"], img_set_config["num_annotations"])

        img_set_config["training_image_paths"] = shuffled[:num_training]
        img_set_config["validation_image_paths"] = shuffled[num_training:(num_training+num_validation)]
        img_set_config["test_image_paths"] = shuffled[(num_training+num_validation):]

    else:
        raise RuntimeError("Missing information needed to create dataset splits.")


    img_set_config["class_map"] = xml_io.create_class_map(img_set_config["all_annotation_paths"])
    img_set_config["num_classes"] = len(img_set_config["class_map"].keys())



    all_img_paths = []
    all_img_paths.extend(img_set_config["training_image_paths"])
    all_img_paths.extend(img_set_config["validation_image_paths"]) 
    all_img_paths.extend(img_set_config["test_image_paths"])


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
