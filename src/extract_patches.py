import logging
import os
import tqdm
import random
import math as m
import numpy as np
import cv2
import uuid


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



def get_patch_coords_surrounding_box(box, patch_size, img_h, img_w):

    logger = logging.getLogger(__name__)

    box_y_min, box_x_min, box_y_max, box_x_max = box

    box_h = box_y_max - box_y_min
    box_w = box_x_max - box_x_min

    if box_h > patch_size or box_w > patch_size:
        logger.warning("Box exceeds size of patch. (box_w, box_h): ({}, {}). patch_size: {}.".format(
            box_w, box_h, patch_size
        ))

    patch_y_min = random.randrange((box_y_min + box_h) - max(box_h, patch_size), box_y_min + 1)
    patch_y_min = min(img_h - patch_size, max(0, patch_y_min))
    patch_x_min = random.randrange((box_x_min + box_w) - max(box_w, patch_size), box_x_min + 1)
    patch_x_min = min(img_w - patch_size, max(0, patch_x_min))
    patch_y_max = patch_y_min + patch_size
    patch_x_max = patch_x_min + patch_size

    patch_coords = [patch_y_min, patch_x_min, patch_y_max, patch_x_max]

    if patch_y_min < 0 or patch_x_min < 0 or patch_y_max > img_h or patch_x_max > img_w:
        raise RuntimeError("Patch exceeds boundaries of the image.")

    return patch_coords


def round_to_multiple(num, multiple):
    return multiple * round(num / multiple)

def extract_patches_from_annotation_guides(image, patch_size, image_annotations, image_annotation_guides):

    annotation_boxes = image_annotations["boxes"]
    annotation_classes = image_annotations["classes"]
    image_patches = []

    image_array = image.load_image_array()
    #image_h, image_w = image_array.shape[:2]

    patch_num = 0

    image_path_pieces = image.image_path.split("/")
    farm_name = image_path_pieces[-5]
    field_name = image_path_pieces[-4]
    mission_date = image_path_pieces[-3]

    patch_overlap_percent = 50
    overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

    for annotation_guide in image_annotation_guides:
        #annotation_size = round(min(annotation_guide["width"], annotation_guide["height"]))


        guide_coords = [
            round(annotation_guide["py"]),
            round(annotation_guide["px"]),
            round(annotation_guide["py"]) + round(annotation_guide["height"]),
            round(annotation_guide["px"]) + round(annotation_guide["width"])
        ]
        print("guide_coords", guide_coords)

        #guide_patch_array = image_array[guide_coords[0]: guide_coords[2],
        #                                guide_coords[1]: guide_coords[3]]


        patch_min_y = guide_coords[0]
   

        col_covered = False
        #patch_min_y = 0
        while not col_covered:
            patch_max_y = patch_min_y + patch_size #tile_size
            if patch_max_y >= guide_coords[2]: #h:
                #patch_min_y = h - tile_size
                #patch_max_y = h
                col_covered = True


            row_covered = False
            #patch_min_x = 0
            patch_min_x = guide_coords[1]
            while not row_covered:

                patch_max_x = patch_min_x + patch_size #tile_size
                if patch_max_x >= guide_coords[3]: #w:
                    #patch_min_x = w - tile_size
                    #patch_max_x = w
                    row_covered = True

                patch_data = {}
                patch_data["image_name"] = image.image_name
                patch_data["image_path"] = image.image_path
                patch_data["patch_name"] = farm_name + "-" + field_name + "-" + mission_date + "-" + \
                                            image.image_name + "-" + str(patch_num).zfill(7) + ".png"
                patch_data["patch_coords"] = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]
                patch_array = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                content = image_array[patch_min_y: min(patch_max_y, guide_coords[2]), patch_min_x: min(patch_max_x, guide_coords[3])] 
                patch_array[0: content.shape[0], 0: content.shape[1], :] = content

                patch_data["patch"] = patch_array
            
                annotate_patch(patch_data, annotation_boxes, annotation_classes, clip_coords=guide_coords) #, clip_coords=guide_coords)

                # output_patch(patch_data["patch"], patch_data["patch_abs_boxes"], pred_boxes=[], pred_classes=[], pred_scores=[], 
                #             out_path=os.path.join("usr", "data", "tmp_patch_data_ex", patch_data["patch_name"]))

                image_patches.append(patch_data)
                patch_num += 1


                patch_min_x += (patch_size - overlap_px)

            patch_min_y += (patch_size - overlap_px)

        # patch_data = {}
        # patch_data["image_name"] = image.image_name
        # patch_data["image_path"] = image.image_path
        # patch_data["patch_name"] = farm_name + "-" + field_name + "-" + mission_date + "-" + \
        #                                     image.image_name + "-" + str(patch_num).zfill(7) + ".png"


        # patch_height = round_to_multiple(round(annotation_guide["height"]), patch_size)
        # patch_width = round_to_multiple(round(annotation_guide["width"]), patch_size)

        # patch_array = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)

        # patch_array[:annotation_guide["height"], :annotation_guide["width"]] = guide_patch_array
        # patch_data["patch_coords"] = [
        #     guide_coords[0],
        #     guide_coords[1],
        #     guide_coords[0] + patch_height,
        #     guide_coords[1] + patch_width
        # ]

                    
        # annotate_patch(patch_data, annotation_boxes, annotation_classes, clip_coords=guide_coords)
            

        # # patch_data["patch"] = guide_patch_array
        # # patch_data["patch_coords"] = guide_coords
        # # annotate_patch(patch_data, annotation_boxes, annotation_classes)

        # # print("annotation_guide size", annotation_size)
        # # print("patch_size", patch_size)
        # # if annotation_size >= patch_size:
        # #     diff = annotation_size - patch_size
        # #     inset = round(diff / 2)
        # #     patch_coords = [guide_coords[0] + inset, 
        # #                     guide_coords[1] + inset, 
        # #                     guide_coords[0] + inset + patch_size, 
        # #                     guide_coords[1] + inset + patch_size]
        # #     patch_array = image_array[patch_coords[0]: patch_coords[2],
        # #                               patch_coords[1]: patch_coords[3]]

        # #     patch_data["patch"] = patch_array
        # #     patch_data["patch_coords"] = patch_coords
        # #     annotate_patch(patch_data, annotation_boxes, annotation_classes)

        
        # # else:

        # #     diff = patch_size - annotation_size
        # #     inset = round(diff / 2)

        # #     patch_array = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        # #     patch_array[inset: inset+annotation_size, inset:inset+annotation_size] = guide_patch_array
        # #     patch_data["patch_coords"] = [
        # #                     guide_coords[0] - inset,
        # #                     guide_coords[1] - inset,
        # #                     guide_coords[0] - inset + patch_size,
        # #                     guide_coords[1] - inset + patch_size
        # #     ]
        # #     patch_data["patch"] = patch_array
            
        # #     annotate_patch(patch_data, annotation_boxes, annotation_classes, clip_coords=guide_coords)


        # output_patch(patch_data["patch"], patch_data["patch_abs_boxes"], pred_boxes=[], pred_classes=[], pred_scores=[], 
        #               out_path=os.path.join("usr", "data", "tmp_patch_data_ex", patch_data["patch_name"]))



        # image_patches.append(patch_data)
        # patch_num += 1


    return image_patches




def extract_patch_records_surrounding_annotations(image, patch_size, image_annotations, include_patch_arrays=True):

    annotation_boxes = image_annotations["boxes"]
    annotation_classes = image_annotations["classes"]
    image_patches = []

    if include_patch_arrays:
        image_array = image.load_image_array()
        image_h, image_w = image_array.shape[:2]
    else:
        image_w, image_h = image.get_wh()

    patch_num = 0

    image_path_pieces = image.image_path.split("/")
    farm_name = image_path_pieces[-5]
    field_name = image_path_pieces[-4]
    mission_date = image_path_pieces[-3]

    for box in annotation_boxes:

        patch_coords = get_patch_coords_surrounding_box(box, patch_size, image_h, image_w)
        
        #patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

        patch_data = {}
        if include_patch_arrays:
            #patch_array = image_array[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
            patch_array = image_array[patch_coords[0]: patch_coords[2],
                                        patch_coords[1]: patch_coords[3]]
            patch_data["patch"] = patch_array
        patch_data["image_name"] = image.image_name
        patch_data["image_path"] = image.image_path
        patch_data["patch_name"] = farm_name + "-" + field_name + "-" + mission_date + "-" + \
                                    image.image_name + "-" + str(patch_num).zfill(7) + ".png"
        patch_data["patch_coords"] = patch_coords
        
        #if image_annotations is not None and \
        #    (annotation_status == "completed_for_training" or annotation_status == "completed_for_testing"):
        annotate_patch(patch_data, annotation_boxes, annotation_classes)
        image_patches.append(patch_data)
        patch_num += 1


    return image_patches




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

    image_path_pieces = image.image_path.split("/")
    farm_name = image_path_pieces[-5]
    field_name = image_path_pieces[-4]
    mission_date = image_path_pieces[-3]

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
            patch_data["patch_name"] = farm_name + "-" + field_name + "-" + mission_date + "-" + \
                                       image.image_name + "-" + str(patch_num).zfill(7) + ".png"
            patch_data["patch_coords"] = patch_coords
            
            if image_annotations is not None and \
                (annotation_status == "completed_for_training" or annotation_status == "completed_for_testing"):
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


def get_contained_inds_2(gt_boxes, patch_coords):

    return np.where(
    np.logical_and(
    np.logical_and(gt_boxes[:,1] < patch_coords[3], gt_boxes[:,3] > patch_coords[1]),
    np.logical_and(gt_boxes[:,0] < patch_coords[2], gt_boxes[:,2] > patch_coords[0])
    ))[0]


def annotate_patch(patch_data, gt_boxes, gt_classes, clip_coords=None):

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

        # WHY ARE WE USING CENTRES OF BOXES ????


        if clip_coords is None:
            contained_inds = get_contained_inds_2(gt_boxes, patch_data["patch_coords"])
            #contained_inds = get_contained_inds(centres, patch_data["patch_coords"])
        else:
            contained_inds = get_contained_inds_2(gt_boxes, clip_coords)
            #contained_inds = get_contained_inds(centres, clip_coords)
        contained_boxes = gt_boxes[contained_inds]
        contained_classes = gt_classes[contained_inds]

        if clip_coords is None:
            image_abs_boxes, mask = box_utils.clip_boxes_and_get_small_visibility_mask(
                contained_boxes, patch_coords, min_visibility=0.15)
        else:
            image_abs_boxes, mask = box_utils.clip_boxes_and_get_small_visibility_mask(
                contained_boxes, clip_coords, min_visibility=0.15)

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




def output_patch(patch, gt_boxes, pred_boxes, pred_classes, pred_scores, out_path):
    from models.common import model_vis

    out_array = model_vis.draw_boxes_on_image(patch,
                      pred_boxes,
                      pred_classes,
                      pred_scores,
                      class_map={"plant": 0},
                      gt_boxes=gt_boxes, #None,
                      patch_coords=None,
                      display_class=False,
                      display_score=False)
    cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))