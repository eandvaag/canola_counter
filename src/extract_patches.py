from abc import ABC, abstractmethod
import math as m
import numpy as np
import uuid
import tqdm
import cv2
import os
import shutil

from io_utils import json_io
from io_utils import xml_io
from io_utils import tf_record_io


def get_patches_dir(dataset_name, extractor_name, patch_size, is_annotated, settings):

    annotated = "annotated" if is_annotated else "unannotated"
    dirname = dataset_name + "-" + extractor_name + "-" + str(patch_size) + "-" + annotated
    return os.path.join(settings.detector_patches_dir, dirname)


def parse_patches_dir(patches_dir):
    info = {}
    vals = os.path.basename(patches_dir).split("-")

    info["dataset_name"] = vals[0]
    info["extractor_name"] = vals[1]
    info["patch_size"] = int(vals[2])
    info["is_annotated"] = True if vals[3] == "annotated" else False

    return info



# class DetectorExtractionScenario:

#     def __init__(self, dataset_name, extractor_name, patch_size, is_annotated):
#         self.dataset_name = dataset_name
#         self.extractor_name = extractor_name
#         self.patch_size = patch_size
#         self.is_annotated = is_annotated


#     def scenario_exists(self, settings):

#         detector_scenario_lookup = json_io.load_json(settings.detector_scenario_lookup_path)

#         for scenario_uuid in detector_scenario_lookup.keys():
#             if  extraction_record[scenario_uuid]["dataset_name"] == self.dataset_name and \
#                 extraction_record[scenario_uuid]["extractor_name"] == self.extractor_name and \
#                 extraction_record[scenario_uuid]["patch_size"] == self.patch_size and \
#                 extraction_record[scenario_uuid]["is_annotated"] == self.is_annotated:

#                 return extraction_record["scenario_uuid"]

#         return None

#     # def delete_scenario(self, settings):

#     #     detector_scenario_lookup = json_io.load_json(settings.detector_scenario_lookup_path)

#     #     for i, extraction_record in enumerate(detector_scenario_lookup["scenarios"]):
#     #         if  extraction_record["dataset_name"] == self.dataset_name and \
#     #             extraction_record["extractor_name"] == self.extractor_name and \
#     #             extraction_record["patch_size"] == self.patch_size and \
#     #             extraction_record["is_annotated"] == self.is_annotated:

#     #             break


#     #     scenario_dir = os.path.join(settings.detector_patches_dir, extraction_record["scenario_uuid"])
#     #     shutil.rmtree(scenario_dir)
#     #     del detector_scenario_lookup["scenarios"][i]
#     #     json_io.save_json(settings.detector_scenario_lookup_path, detector_scenario_lookup)




# def update_scenario_lookup(scenario, scenario_uuid, settings):

#     detector_scenario_lookup = json_io.load_json(settings.detector_scenario_lookup_path)

#     detector_scenario_lookup[scenario_uuid] = {
#             "dataset_name": scenario.dataset_name,
#             "extractor_name": scenario.extractor_name,
#             "patch_size": scenario.patch_size,
#             "is_annotated": scenario.is_annotated
#     }

#     json_io.save_json(settings.detector_scenario_lookup_path, detector_scenario_lookup)


def write_patches(out_dir, patch_data):

    for patch, patch_name in zip(patch_data["patches"], patch_data["patch_names"]):
        cv2.imwrite(os.path.join(out_dir, patch_name), patch)



class DetectorPatchExtractor(ABC):

    @property
    def name(self):
        pass


    def extract(self, dataset, settings, annotate_patches):

        #scenario = DetectorExtractionScenario(dataset.name, self.name, settings.detector_patch_size, annotate_patches)

        patches_dir = get_patches_dir(dataset.name, self.name, settings.detector_patch_size, annotate_patches, settings)

        if os.path.exists(patches_dir):
            return patches_dir #os.path.join(settings.detector_patches_dir, scenario_uuid, "record.tfrec")

        #scenario_uuid = str(uuid.uuid4())
        #out_dir = os.path.join(settings.detector_patches_dir, scenario_uuid)
        os.makedirs(patches_dir)
        patch_tf_records = []

        for img in tqdm.tqdm(dataset.imgs, desc="Generating patches"):

            if annotate_patches and not img.is_annotated:
                continue

            patch_data = self._extract_patches_from_img(img, settings)
            #print("patch_coords", patch_data["patch_coords"])
            #exit()
            write_patches(patches_dir, patch_data)

            if annotate_patches:
                gt_boxes, gt_classes = xml_io.load_boxes_and_classes(img.xml_path, settings.class_map)

                patch_data = annotate_patch_data(img, patch_data, gt_boxes, gt_classes)

            patch_tf_records_for_img = tf_record_io.create_patch_tf_records_for_img(img, patch_data, patches_dir, annotate_patches)

            patch_tf_records.extend(patch_tf_records_for_img)


        tf_record_io.output_patch_tf_records(patches_dir, patch_tf_records)
        #update_scenario_lookup(scenario, scenario_uuid, settings)

        return patches_dir #os.path.join(out_dir, "record.tfrec")


    def delete_scenario(self, dataset, settings, annotate_patches):

        scenario = DetectorExtractionScenario(dataset.name, self.name, settings.detector_patch_size, annotate_patches)
        scenario.delete_scenario(settings)


    @abstractmethod
    def _extract_patches_from_img(self, img, settings):
        pass



def get_scenario_information(scenario_uuid, settings):

    detector_scenario_lookup = json_io.load_json(settings.detector_scenario_lookup_path)
    return detector_scenario_lookup[scenario_uuid]


def get_scenario_record_path(scenario_uuid, settings):

    scenario_info = get_scenario_information(scenario_uuid)
    return os.path.join(settings.detector_patches_dir, scenario_uuid, "record.tfrec")


class BoxExtractor(DetectorPatchExtractor):

    @property
    def name(self):
        return "box"


    def _extract_patches_from_img(self, img, settings):

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
        for centre in centres:
            try:
                patch, patch_coords = extract_patch_surrounding_point(img_array, centre, settings.detector_patch_size)
                patch_data["patches"].append(patch)
                patch_data["patch_coords"].append(patch_coords)
                patch_data["patch_names"].append(os.path.basename(img.img_path[:-4]) + "-patch-" + str(patch_num).zfill(5) + ".png")
                patch_num += 1
            except RuntimeError as e:
                print(e)

        return patch_data



    
def extract_patch_surrounding_point(img_array, point, patch_size):
    """
    :param point: (y, x) format
    """

    h, w = img_array.shape[:2]

    y = point[0]
    x = point[1]

    y_min = y - (patch_size // 2)
    x_min = x - (patch_size // 2)
    y_max = y + (patch_size // 2)
    x_max = x + (patch_size // 2)

    patch_coords = [y_min, x_min, y_max, x_max]

    if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
        raise RuntimeError("Patch exceeds boundaries of the image")

    patch = img_array[y_min: y_max, x_min: x_max]

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







# def output_detector_patch_record(dataset, patch_data, extraction_method, settings, is_annotated):

#     dataset_name = dataset.name
#     patch_size = patch_data["patches"][0].shape[0]
#     extraction_method = extraction_method
#     is_annotated = is_annotated

#     record_lookup_path = settings.detector_patch_lookup

#     entry_exists_in_detector_record_lookup(record_lookup_path, dataset_name, patch_size, extraction_method, is_annotated)
#     uuid = add_entry_to_detector_record_lookup(patch_data, settings)



def annotate_patch_data(img, patch_data, gt_boxes, gt_classes):

    def get_contained_inds(centres, patch_coords):
        return np.where(np.logical_and(
                            np.logical_and(centres[:,0] > patch_coords[0], 
                                           centres[:,0] < patch_coords[2]),
                            np.logical_and(centres[:,1] > patch_coords[1], 
                                           centres[:,1] < patch_coords[3])))[0]



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


    patch_num = 0
    for patch, patch_coords in zip(patch_data["patches"], patch_data["patch_coords"]):

        contained_inds = get_contained_inds(centres, patch_coords)
        contained_boxes = gt_boxes[contained_inds]
        contained_classes = gt_classes[contained_inds]

        if contained_boxes.size == 0:
            continue


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
        annotated_patch_data["patch_names"].append(os.path.basename(img.img_path[:-4]) + "-patch-" + str(patch_num).zfill(5) + ".png")
        annotated_patch_data["img_abs_boxes"].append(img_abs_boxes.tolist())
        annotated_patch_data["patch_abs_boxes"].append(patch_abs_boxes.tolist())
        annotated_patch_data["patch_normalized_boxes"].append(patch_normalized_boxes.tolist())
        annotated_patch_data["patch_classes"].append(contained_classes.tolist())
        patch_num += 1


    return annotated_patch_data



