from abc import ABC, abstractmethod
import math as m
import random
import numpy as np
import uuid
import tqdm
import cv2
import os
import shutil
import logging

from io_utils import json_io
from io_utils import xml_io
from io_utils import tf_record_io


def get_patch_dir(img_set, dataset_name, extractor_name, patch_size, is_annotated):

    annotated = "annotated" if is_annotated else "unannotated"
    dirname = dataset_name + "-" + extractor_name + "-" + str(patch_size) + "-" + annotated
    return os.path.join(img_set.patch_dir, dirname)


def parse_patch_dir(patch_dir):
    info = {}
    vals = os.path.basename(patch_dir).split("-")

    info["dataset_name"] = vals[0]
    info["extractor_name"] = vals[1]
    info["patch_size"] = int(vals[2])
    info["is_annotated"] = True if vals[3] == "annotated" else False

    return info



def write_patches(out_dir, patch_data):

    for patch, patch_name in zip(patch_data["patches"], patch_data["patch_names"]):
        cv2.imwrite(os.path.join(out_dir, patch_name), patch)



def search_for_scenario(img_set, scenario):

    scenario_lookup_path = os.path.join(img_set.patch_dir, "scenario_lookup.json")
    scenario_lookup = json_io.load_json(scenario_lookup_path)

    for scenario_id, stored_scenario in scenario_lookup["scenarios"].items():
        
        if scenario == stored_scenario:
            return True, scenario_id

    return False, None


def generate_scenario_id():
    return str(uuid.uuid4())


def add_scenario(img_set, scenario_id, scenario):

    scenario_lookup_path = os.path.join(img_set.patch_dir, "scenario_lookup.json")
    scenario_lookup = json_io.load_json(scenario_lookup_path)

    scenario_lookup["scenarios"][scenario_id] = scenario

    json_io.save_json(scenario_lookup_path, scenario_lookup)


def extract_patches(params, img_set, dataset, annotate_patches):

    scenario = {"dataset_name": dataset.name,
                "parameters":   params,
                "is_annotated": annotate_patches}


    scenario_exists, scenario_id = search_for_scenario(img_set, scenario)
    if scenario_exists:
        return scenario_id
    else:
        scenario_id = generate_scenario_id()


    if params["method"] == "tile":
        extractor = TileExtractor(params)
    elif params["method"] == "box":
        extractor = BoxExtractor(params)
    elif params["method"] == "jitterbox":
        extractor = JitterBoxExtractor(params)
    else:
        raise RuntimeError("Unrecognized patch extraction method: '{}'.".format(params["method"]))

    extractor.extract(scenario_id, scenario, img_set)
    return scenario_id


class DetectorPatchExtractor(ABC):

    @property
    def name(self):
        pass

    def __init__(self, params):
        self.patch_size = params["patch_size"]


    def extract(self, scenario_id, scenario, img_set):

        #patch_dir = get_patch_dir(img_set, dataset.name, self.name, self.patch_size, annotate_patches)
        annotate_patches = scenario["is_annotated"]
        dataset = img_set.datasets[scenario["dataset_name"]]


        patch_dir = os.path.join(img_set.patch_dir, scenario_id)
        patches_record_path = os.path.join(patch_dir, "patches-record.tfrec")
        patches_with_boxes_record_path = os.path.join(patch_dir, "patches-with-boxes-record.tfrec")

        #if os.path.exists(patches_record_path) and (os.path.exists(patches_with_boxes_record_path) or not annotate_patches):
        #    return patch_dir

        #if os.path.exists(patch_dir):
        #    shutil.rmtree(patch_dir)

        os.makedirs(patch_dir)

        tf_records = []
        tf_records_with_boxes = []

        for img in tqdm.tqdm(dataset.imgs, desc="Generating patches"):

            if annotate_patches and not img.is_annotated:
                continue

            patch_data = self._extract_patches_from_img(img, img_set)

            write_patches(patch_dir, patch_data)

            if annotate_patches:
                gt_boxes, gt_classes = xml_io.load_boxes_and_classes(img.xml_path, img_set.class_map)

                patch_data = annotate_patch_data(img, patch_data, gt_boxes, gt_classes)

                tf_records_with_boxes_for_img = tf_record_io.create_patch_tf_records_for_img(img, patch_data, 
                                                                                             patch_dir, annotate_patches, 
                                                                                             require_box=True)


            tf_records_for_img = tf_record_io.create_patch_tf_records_for_img(img, patch_data, patch_dir, annotate_patches)



            tf_records.extend(tf_records_for_img)
            tf_records_with_boxes.extend(tf_records_with_boxes_for_img)


        tf_record_io.output_patch_tf_records(patches_record_path, tf_records)
        tf_record_io.output_patch_tf_records(patches_with_boxes_record_path, tf_records_with_boxes)


        add_scenario(img_set, scenario_id, scenario)


    @abstractmethod
    def _extract_patches_from_img(self, img, img_set):
        pass




class JitterBoxExtractor(DetectorPatchExtractor):

    @property
    def name(self):
        return "jitterbox"

    def __init__(self, params):
        super().__init__(params)
        self.num_patches_per_box = params["num_patches_per_box"]

    def _extract_patches_from_img(self, img, img_set):

        logger = logging.getLogger(__name__)

        patch_data = {
            "patches": [],
            "patch_coords": [],
            "patch_names": [],
        }

        img_array = img.load_img_array()
        gt_boxes, _ = xml_io.load_boxes_and_classes(img.xml_path, img_set.class_map)

        centres = np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)

        patch_num = 0
        num_excluded = 0
        for gt_box in gt_boxes:
            for i in range(self.num_patches_per_box):
                try:
                    patch, patch_coords = self._extract_patch_surrounding_gt_box(img_array, gt_box, self.patch_size)
                    patch_data["patches"].append(patch)
                    patch_data["patch_coords"].append(patch_coords)
                    patch_data["patch_names"].append(os.path.basename(img.img_path[:-4]) + "-patch-" + str(patch_num).zfill(5) + ".png")
                    patch_num += 1
                except RuntimeError as e:
                    num_excluded += 1
        
        num_extracted = len(patch_data["patch_names"])

        if num_excluded > 0:
            logger.info("JitterBoxExtractor: Some patches were excluded because they did not fall " +
                        "within the boundaries of the image.")

        logger.info("JitterBoxExtractor: Extracted {} patches ({} attempts.)".format(num_extracted, num_extracted + num_excluded))

        return patch_data



    def _extract_patch_surrounding_gt_box(self, img_array, gt_box, patch_size):

        img_h, img_w = img_array.shape[:2]

        box_y_min, box_x_min, box_y_max, box_x_max = gt_box

        box_h = box_y_max - box_y_min
        box_w = box_x_max - box_x_min

        if box_h > patch_size or box_w > patch_size:
            raise RuntimeError("Box exceeds size of patch.")

        patch_y_min = random.randrange((box_y_min + box_h) - patch_size, box_y_min)  
        patch_x_min = random.randrange((box_x_min + box_w) - patch_size, box_x_min)
        patch_y_max = patch_y_min + patch_size
        patch_x_max = patch_x_min + patch_size

        patch_coords = [patch_y_min, patch_x_min, patch_y_max, patch_x_max]

        if patch_y_min < 0 or patch_x_min < 0 or patch_y_max > img_h or patch_x_max > img_w:
            raise RuntimeError("Patch exceeds boundaries of the image.")

        patch = img_array[patch_y_min:patch_y_max, patch_x_min:patch_x_max]

        return patch, patch_coords



class BoxExtractor(DetectorPatchExtractor):

    @property
    def name(self):
        return "box"

    def _extract_patches_from_img(self, img, img_set):

        logger = logging.getLogger(__name__)

        patch_data = {
            "patches": [],
            "patch_coords": [],
            "patch_names": [],
        }

        img_array = img.load_img_array()
        gt_boxes, _ = xml_io.load_boxes_and_classes(img.xml_path, img_set.class_map)
        
        centres = np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)

        patch_num = 0
        num_excluded = 0
        for centre in centres:
            try:
                patch, patch_coords = self._extract_patch_surrounding_point(img_array, centre, self.patch_size)
                patch_data["patches"].append(patch)
                patch_data["patch_coords"].append(patch_coords)
                patch_data["patch_names"].append(os.path.basename(img.img_path[:-4]) + "-patch-" + str(patch_num).zfill(5) + ".png")
                patch_num += 1
            except RuntimeError as e:
                num_excluded += 1
        
        logger.info("BoxExtractor: Excluded {} patches that exceeded the boundaries of the image.".format(num_excluded))

        return patch_data



    
    def _extract_patch_surrounding_point(self, img_array, point, patch_size):
        """
        :param point: (y, x) format
        """

        img_h, img_w = img_array.shape[:2]

        y = point[0]
        x = point[1]

        patch_y_min = y - (patch_size // 2)
        patch_x_min = x - (patch_size // 2)
        patch_y_max = y + (patch_size // 2)
        patch_x_max = x + (patch_size // 2)

        patch_coords = [patch_y_min, patch_x_min, patch_y_max, patch_x_max]

        if patch_y_min < 0 or patch_x_min < 0 or patch_y_max > img_h or patch_x_max > img_w:
            raise RuntimeError("Patch exceeds boundaries of the image.")

        patch = img_array[patch_y_min: patch_y_max, patch_x_min: patch_x_max]

        return patch, patch_coords


class TileExtractor(DetectorPatchExtractor):

    @property
    def name(self):
        return 'tile'


    def __init__(self, params):
        super().__init__(params)
        self.patch_overlap_pct = params["patch_overlap_pct"]

    def _extract_patches_from_img(self, img, img_set):

        patch_data = {
            "patches": [],
            "patch_coords": [],
            "patch_names": [],
        }


        img_array = img.load_img_array()
        img_path = img.img_path

        tile_size = self.patch_size
        overlap_px = int(m.floor(tile_size * (self.patch_overlap_pct / 100)))

        h, w = img_array.shape[:2]
        patch_num = 0
        for row_ind in range(0, h, tile_size - overlap_px):
            for col_ind in range(0, w, tile_size - overlap_px):

                patch_min_y = row_ind
                patch_min_x = col_ind

                patch_max_y = row_ind + tile_size
                if patch_max_y > h:
                    patch_min_y = h - tile_size
                    patch_max_y = h

                patch_max_x = col_ind + tile_size
                if patch_max_x > w:
                    patch_min_x = w - tile_size
                    patch_max_x = w


                patch = img_array[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
                patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                patch_data["patches"].append(patch)
                patch_data["patch_coords"].append(patch_coords)
                patch_data["patch_names"].append(os.path.basename(img.img_path[:-4]) + "-patch-" + str(patch_num).zfill(5) + ".png")
                patch_num += 1

        return patch_data





def get_contained_inds(centres, patch_coords):
    return np.where(np.logical_and(
                        np.logical_and(centres[:,0] > patch_coords[0], 
                                       centres[:,0] < patch_coords[2]),
                        np.logical_and(centres[:,1] > patch_coords[1], 
                                       centres[:,1] < patch_coords[3])))[0]


def annotate_patch_data(img, patch_data, gt_boxes, gt_classes):




    annotated_patch_data = {
        "patches": [],
        "patch_coords": [],
        "patch_names": [],
        "img_abs_boxes": [],
        "patch_abs_boxes": [],
        "patch_normalized_boxes": [],
        "patch_classes": []
    }

    centres =  np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)


    for i in range(len(patch_data["patches"])):

        patch = patch_data["patches"][i]
        patch_coords = patch_data["patch_coords"][i]
        patch_name = patch_data["patch_names"][i]

        contained_inds = get_contained_inds(centres, patch_coords)
        contained_boxes = gt_boxes[contained_inds]
        contained_classes = gt_classes[contained_inds]

        #if contained_boxes.size == 0:
        #    continue


        # boxes are clipped to be contained within the patch
        img_abs_boxes = np.stack([np.maximum(contained_boxes[:,0], patch_coords[0]),
                                  np.maximum(contained_boxes[:,1], patch_coords[1]),
                                  np.minimum(contained_boxes[:,2], patch_coords[2]),
                                  np.minimum(contained_boxes[:,3], patch_coords[3])], axis=-1)



        patch_abs_boxes = np.stack([img_abs_boxes[:,0] - patch_coords[0],
                                    img_abs_boxes[:,1] - patch_coords[1],
                                    img_abs_boxes[:,2] - patch_coords[0],
                                    img_abs_boxes[:,3] - patch_coords[1]], axis=-1)


        patch_normalized_boxes = patch_abs_boxes / patch.shape[0]


        annotated_patch_data["patches"].append(patch)
        annotated_patch_data["patch_coords"].append(patch_coords)
        annotated_patch_data["patch_names"].append(patch_name)
        annotated_patch_data["img_abs_boxes"].append(img_abs_boxes.tolist())
        annotated_patch_data["patch_abs_boxes"].append(patch_abs_boxes.tolist())
        annotated_patch_data["patch_normalized_boxes"].append(patch_normalized_boxes.tolist())
        annotated_patch_data["patch_classes"].append(contained_classes.tolist())

    return annotated_patch_data



