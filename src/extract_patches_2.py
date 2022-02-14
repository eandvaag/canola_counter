import os
import shutil
import tqdm

import math as m
import numpy as np

import cv2
import uuid

from io_utils import w3c_io, tf_record_io
from models.common import box_utils


def extract_patches(dataset, config):


    annotations = w3c_io.load_annotations(dataset.annotations_path, config.arch["class_map"])

    patch_dir = os.path.join(config.model_dir, "patches", str(uuid.uuid4()))
    os.makedirs(patch_dir)
    annotated_patches_record_path = os.path.join(patch_dir, "annotated-patches-record.tfrec")
    annotated_patches_with_boxes_record_path = os.path.join(patch_dir, "annotated-patches-with-boxes-record.tfrec")
    annotated_patches_with_no_boxes_record_path = os.path.join(patch_dir, "annotated-patches-with-no-boxes-record.tfrec")
    unannotated_patches_record_path = os.path.join(patch_dir, "unannotated-patches-record.tfrec")
    print("writing patches to {} and {}".format(annotated_patches_record_path, unannotated_patches_record_path))
    

    annotated_tf_records = []
    annotated_tf_records_with_boxes = []
    annotated_tf_records_with_no_boxes = []
    unannotated_tf_records = []

    for image in tqdm.tqdm(dataset.images, desc="Generating patches"):
        
        is_annotated = annotations[image.image_name]["status"] == "completed"
        image_patches = extract_patches_from_image(image, dataset.patch_extraction_params, annotations[image.image_name])
                                                 #boxes=boxes[image.image_name], classes=classes[image.image_name])

        write_patches(patch_dir, image_patches)
        tf_records_for_image = tf_record_io.create_patch_tf_records_for_image(image, image_patches, patch_dir, is_annotated=is_annotated)
        


        if is_annotated:
            annotated_tf_records.extend(tf_records_for_image)


            tf_records_for_image_with_boxes = tf_record_io.create_patch_tf_records_for_image(image, image_patches, 
                                                                                             patch_dir, is_annotated, 
                                                                                             require_box=True)

            tf_records_for_image_with_no_boxes = tf_record_io.create_patch_tf_records_for_image(image, image_patches, 
                                                                                             patch_dir, is_annotated, 
                                                                                             require_no_box=True)

            annotated_tf_records_with_boxes.extend(tf_records_for_image_with_boxes) 
            annotated_tf_records_with_no_boxes.extend(tf_records_for_image_with_no_boxes)  


        else:
            unannotated_tf_records.extend(tf_records_for_image)


    tf_record_io.output_patch_tf_records(annotated_patches_record_path, annotated_tf_records)
    tf_record_io.output_patch_tf_records(annotated_patches_with_boxes_record_path, annotated_tf_records_with_boxes)
    tf_record_io.output_patch_tf_records(annotated_patches_with_no_boxes_record_path, annotated_tf_records_with_no_boxes)
    tf_record_io.output_patch_tf_records(unannotated_patches_record_path, unannotated_tf_records)


    return patch_dir

def extract_patches_for_graph(dataset, patch_dir):

    if os.path.exists(patch_dir):
       shutil.rmtree(patch_dir)
    os.makedirs(patch_dir) 

    annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})

    #patch_dir = os.path.join(config.model_dir, "patches", str(uuid.uuid4()))
    #os.makedirs(patch_dir)
    #patches_record_path = os.path.join(patch_dir, "patches-record.tfrec")
    #print("writing patches to ", patches_record_path)
    

    #tf_records = []

    for image in tqdm.tqdm(dataset.images, desc="Generating patches"):
        
        #image_patches = extract_patches_from_image(image, dataset.patch_extraction_params, annotations[image.image_name]) #boxes=None, classes=None) #boxes[image.image_name], classes=classes[image.image_name])

        image_patches = extract_gt_boxes(image, annotations[image.image_name]["boxes"])

        write_patches(patch_dir, image_patches)
        #tf_records_for_image = tf_record_io.create_patch_tf_records_for_image(image, image_patches, patch_dir, is_annotated=True)
        #tf_records.extend(tf_records_for_image)


    #tf_record_io.output_patch_tf_records(patches_record_path, tf_records)


    #return patch_dir


def extract_gt_boxes(image, gt_boxes):

    image_array = image.load_image_array()

    patch_num = 0
    gt_patches = []
    for gt_box in gt_boxes:
        gt_patch = image_array[gt_box[0]:gt_box[2], gt_box[1]:gt_box[3]]
        patch_data = {}
        patch_data["patch"] = gt_patch
        patch_data["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
        gt_patches.append(patch_data)
        patch_num += 1
    return gt_patches



def write_patches(out_dir, patch_data_lst):
    for patch_data in patch_data_lst:
        cv2.imwrite(os.path.join(out_dir, patch_data["patch_name"]),
                    cv2.cvtColor(patch_data["patch"], cv2.COLOR_RGB2BGR))


def extract_patches_from_image(image, patch_extraction_params, image_annotations):

    method = patch_extraction_params["method"]
    if method == "tile":
        image_patches = extract_patches_from_image_tile(image, patch_extraction_params, image_annotations)
    else:
        raise RuntimeError("Unsupported patch extaction method")
    
    return image_patches


def extract_patches_from_image_tile(image, patch_extraction_params, image_annotations): #boxes=None, classes=None):


    patch_size = patch_extraction_params["patch_size"]
    patch_overlap_percent = patch_extraction_params["patch_overlap_percent"]

    annotation_status = image_annotations["status"]
    annotation_boxes = image_annotations["boxes"]
    annotation_classes = image_annotations["classes"]

    image_patches = []

    image_array = image.load_image_array()
    image_path = image.image_path

    tile_size = patch_size
    overlap_px = int(m.floor(tile_size * (patch_overlap_percent / 100)))

    h, w = image_array.shape[:2]
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


            patch = image_array[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
            patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

            patch_data = {}
            patch_data["patch"] = patch
            patch_data["patch_coords"] = patch_coords
            patch_data["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
            #if annotate_patches:
            if annotation_status == "completed":
                annotate_patch(patch_data, annotation_boxes, annotation_classes)
            image_patches.append(patch_data)
            patch_num += 1

    return image_patches



def get_contained_inds(centres, patch_coords):
    return np.where(np.logical_and(
                        np.logical_and(centres[:,0] > patch_coords[0], 
                                       centres[:,0] < patch_coords[2]),
                        np.logical_and(centres[:,1] > patch_coords[1], 
                                       centres[:,1] < patch_coords[3])))[0]


def annotate_patch(patch_data, gt_boxes, gt_classes):

    patch = patch_data["patch"]
    patch_coords = patch_data["patch_coords"]

    centres =  np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)
    contained_inds = get_contained_inds(centres, patch_data["patch_coords"])
    contained_boxes = gt_boxes[contained_inds]
    contained_classes = gt_classes[contained_inds]

    patch_height = patch_coords[2] - patch_coords[0]
    patch_width = patch_coords[3] - patch_coords[1]
    image_abs_boxes = box_utils.clip_and_remove_small_visibility_boxes_np(
        contained_boxes, patch_coords, min_visibility=0)

    #image_abs_boxes = box_utils.clip_boxes_np(contained_boxes, patch_coords)

    # # boxes are clipped to be contained within the patch
    # image_abs_boxes = np.stack([np.maximum(contained_boxes[:,0], patch_coords[0]),
    #                           np.maximum(contained_boxes[:,1], patch_coords[1]),
    #                           np.minimum(contained_boxes[:,2], patch_coords[2]),
    #                           np.minimum(contained_boxes[:,3], patch_coords[3])], axis=-1)

    patch_abs_boxes = np.stack([image_abs_boxes[:,0] - patch_coords[0],
                                image_abs_boxes[:,1] - patch_coords[1],
                                image_abs_boxes[:,2] - patch_coords[0],
                                image_abs_boxes[:,3] - patch_coords[1]], axis=-1)

    patch_normalized_boxes = patch_abs_boxes / patch.shape[0]

    patch_data["image_abs_boxes"] = image_abs_boxes.tolist()
    patch_data["patch_abs_boxes"] = patch_abs_boxes.tolist()
    patch_data["patch_normalized_boxes"] = patch_normalized_boxes.tolist()
    patch_data["patch_classes"] = contained_classes.tolist()

    return patch_data