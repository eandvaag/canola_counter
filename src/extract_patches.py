
import os
import tqdm
import math as m
import numpy as np
import cv2


from models.common import box_utils
from io_utils import tf_record_io



def write_annotated_patch_records(patch_records, patch_dir, includes_patch_arrays=True):

    if includes_patch_arrays:
        write_patches(patch_dir, patch_records)
    annotated_tf_records = tf_record_io.create_patch_tf_records(patch_records, patch_dir, is_annotated=True)
    annotated_patches_record_path = os.path.join(patch_dir, "annotated-patches-record.tfrec")
    tf_record_io.output_patch_tf_records(annotated_patches_record_path, annotated_tf_records)



def write_patches(out_dir, patch_records):
    for patch_record in tqdm.tqdm(patch_records, desc="Writing patches"):
        cv2.imwrite(os.path.join(out_dir, patch_record["patch_name"]),
                    cv2.cvtColor(patch_record["patch"], cv2.COLOR_RGB2BGR))


def add_annotations_to_patch_records(patch_records, image_annotations):
    annotation_boxes = image_annotations["boxes"]
    annotation_classes = image_annotations["classes"]

    for patch_record in patch_records:
        annotate_patch(patch_record, annotation_boxes, annotation_classes)



def extract_patch_records_from_image_tiled(image, patch_size, image_annotations=None, 
                                           patch_overlap_percent=50, include_patch_arrays=True):

    if image_annotations is not None:
        annotation_status = image_annotations["status"]
        annotation_boxes = image_annotations["boxes"]
        annotation_classes = image_annotations["classes"]

    image_patches = []

    if include_patch_arrays:
        image_array = image.load_image_array()
        h, w = image_array.shape[:2]
    else:
        w, h = image.get_wh()

    tile_size = patch_size
    overlap_px = int(m.floor(tile_size * (patch_overlap_percent / 100)))

    
    patch_num = 0


    col_covered = False
    patch_min_y = 0
    while not col_covered:
        patch_max_y = patch_min_y + tile_size
        if patch_max_y >= h:
            patch_min_y = h - tile_size
            patch_max_y = h
            col_covered = True

        row_covered = False
        patch_min_x = 0
        while not row_covered:

            patch_max_x = patch_min_x + tile_size
            if patch_max_x >= w:
                patch_min_x = w - tile_size
                patch_max_x = w
                row_covered = True

            
            patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

            patch_data = {}
            if include_patch_arrays:
                patch_array = image_array[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
                patch_data["patch"] = patch_array
            patch_data["image_name"] = image.image_name
            patch_data["image_path"] = image.image_path
            patch_data["patch_name"] = image.image_name + "-" + str(patch_num).zfill(7) + ".png"
            patch_data["patch_coords"] = patch_coords
            
            if image_annotations is not None and annotation_status == "completed":
                annotate_patch(patch_data, annotation_boxes, annotation_classes)
            image_patches.append(patch_data)
            patch_num += 1
            
            patch_min_x += (tile_size - overlap_px)

        patch_min_y += (tile_size - overlap_px)

    return image_patches





def get_contained_inds(centres, patch_coords):
    return np.where(np.logical_and(
                        np.logical_and(centres[:,0] > patch_coords[0], 
                                        centres[:,0] < patch_coords[2]),
                        np.logical_and(centres[:,1] > patch_coords[1], 
                                        centres[:,1] < patch_coords[3])))[0]




def annotate_patch(patch_data, gt_boxes, gt_classes):

    if gt_boxes.size == 0:
        patch_data["image_abs_boxes"] = []
        patch_data["patch_abs_boxes"] = []
        patch_data["patch_normalized_boxes"] = []
        patch_data["patch_classes"] = [] 

    else:

        #patch = patch_data["patch"]
        patch_coords = patch_data["patch_coords"] 

        patch_size = patch_coords[2] - patch_coords[0]

        centres = np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)
        contained_inds = get_contained_inds(centres, patch_data["patch_coords"])
        contained_boxes = gt_boxes[contained_inds]
        contained_classes = gt_classes[contained_inds]

        image_abs_boxes, mask = box_utils.clip_boxes_and_get_small_visibility_mask(
            contained_boxes, patch_coords, min_visibility=0.05)

        image_abs_boxes = image_abs_boxes[mask]
        contained_classes = contained_classes[mask]

        patch_abs_boxes = np.stack([image_abs_boxes[:,0] - patch_coords[0],
                                    image_abs_boxes[:,1] - patch_coords[1],
                                    image_abs_boxes[:,2] - patch_coords[0],
                                    image_abs_boxes[:,3] - patch_coords[1]], axis=-1)

        patch_normalized_boxes = patch_abs_boxes / patch_size

        patch_data["image_abs_boxes"] = image_abs_boxes.tolist()
        patch_data["patch_abs_boxes"] = patch_abs_boxes.tolist()
        patch_data["patch_normalized_boxes"] = patch_normalized_boxes.tolist()
        patch_data["patch_classes"] = contained_classes.tolist()

    return patch_data