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


def get_patch_dir(dataset_name, extractor_name, patch_size, is_annotated, settings):

    annotated = "annotated" if is_annotated else "unannotated"
    dirname = dataset_name + "-" + extractor_name + "-" + str(patch_size) + "-" + annotated
    return os.path.join(settings.detector_patch_dir, dirname)


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



class DetectorPatchExtractor(ABC):

    @property
    def name(self):
        pass


    def extract(self, dataset, settings, annotate_patches):

        patch_dir = get_patch_dir(dataset.name, self.name, settings.detector_patch_size, annotate_patches, settings)
        patches_record_path = os.path.join(patch_dir, "patches-record.tfrec")
        patches_with_boxes_record_path = os.path.join(patch_dir, "patches-with-boxes-record.tfrec")


        if os.path.exists(patches_record_path) and (os.path.exists(patches_with_boxes_record_path) or not annotate_patches):
            return patch_dir

        if os.path.exists(patch_dir):
            shutil.rmtree(patch_dir)
            
        os.makedirs(patch_dir)
        tf_records = []
        tf_records_with_boxes = []

        for img in tqdm.tqdm(dataset.imgs, desc="Generating patches"):

            if annotate_patches and not img.is_annotated:
                continue

            patch_data = self._extract_patches_from_img(img, settings)

            write_patches(patch_dir, patch_data)

            if annotate_patches:
                gt_boxes, gt_classes = xml_io.load_boxes_and_classes(img.xml_path, settings.class_map)

                patch_data = annotate_patch_data(img, patch_data, gt_boxes, gt_classes)

                tf_records_with_boxes_for_img = tf_record_io.create_patch_tf_records_for_img(img, patch_data, 
                                                                                             patch_dir, annotate_patches, 
                                                                                             require_box=True)


            tf_records_for_img = tf_record_io.create_patch_tf_records_for_img(img, patch_data, patch_dir, annotate_patches)



            tf_records.extend(tf_records_for_img)
            tf_records_with_boxes.extend(tf_records_with_boxes_for_img)


        tf_record_io.output_patch_tf_records(patches_record_path, tf_records)
        tf_record_io.output_patch_tf_records(patches_with_boxes_record_path, tf_records_with_boxes)

        return patch_dir


    def delete_scenario(self, dataset, settings, annotate_patches):

        scenario = DetectorExtractionScenario(dataset.name, self.name, settings.detector_patch_size, annotate_patches)
        scenario.delete_scenario(settings)


    @abstractmethod
    def _extract_patches_from_img(self, img, settings):
        pass




class JitterBoxExtractor(DetectorPatchExtractor):

    @property
    def name(self):
        return "jitterbox"


    def _extract_patches_from_img(self, img, settings):

        logger = logging.getLogger(__name__)

        patch_data = {
            "patches": [],
            "patch_coords": [],
            "patch_names": [],
        }

        img_array = img.load_img_array()
        gt_boxes, _ = xml_io.load_boxes_and_classes(img.xml_path, settings.class_map)

        gt_boxes = np.array(gt_boxes)
        centres = np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)

        patch_num = 0
        num_excluded = 0
        for gt_box in gt_boxes:
            try:
                patch, patch_coords = self._extract_patch_surrounding_gt_box(img_array, gt_box, settings.detector_patch_size)
                patch_data["patches"].append(patch)
                patch_data["patch_coords"].append(patch_coords)
                patch_data["patch_names"].append(os.path.basename(img.img_path[:-4]) + "-patch-" + str(patch_num).zfill(5) + ".png")
                patch_num += 1
            except RuntimeError as e:
                num_excluded += 1
        
        logger.info("JitterBoxExtractor: Excluded {} patches that exceeded the boundaries of the image.".format(num_excluded))

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


    def _extract_patches_from_img(self, img, settings):

        logger = logging.getLogger(__name__)

        patch_data = {
            "patches": [],
            "patch_coords": [],
            "patch_names": [],
        }

        img_array = img.load_img_array()
        gt_boxes, _ = xml_io.load_boxes_and_classes(img.xml_path, settings.class_map)

        gt_boxes = np.array(gt_boxes)
        centres = np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)

        patch_num = 0
        num_excluded = 0
        for centre in centres:
            try:
                patch, patch_coords = self._extract_patch_surrounding_point(img_array, centre, settings.detector_patch_size)
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


    def _extract_patches_from_img(self, img, settings):

        patch_data = {
            "patches": [],
            "patch_coords": [],
            "patch_names": [],
        }


        img_array = img.load_img_array()
        img_path = img.img_path

        tile_size = settings.detector_patch_size
        overlap_px = int(m.floor(tile_size * settings.detector_tile_patch_overlap))

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



