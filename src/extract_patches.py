import os
import glob
import shutil
import tqdm

import math as m
import random
import numpy as np
from functools import reduce

import cv2
import uuid
import logging

from io_utils import w3c_io, tf_record_io
from models.common import box_utils
from image_set import Image, DataSet
import image_utils


from graph import graph_match





# def extract_source_patches(config):
    
#     source_construction_method = config.training["source_construction_params"]["method"]

#     if source_construction_method == "graph_subset":
#         graph_match.create_graph_subset(config)

#     elif source_construction_method == "random_subset":
#         #create_random_subset(model_dir, job_config)
#         pass
    
#     elif source_construction_method == "even_subset":
#         create_even_subset(config)
#     #elif source_construction_method == "target_set":

#     else:
#         raise RuntimeError("Unknown source construction method specified")



# def extract_patches(dataset, config):


#     annotations = w3c_io.load_annotations(dataset.annotations_path, config.arch["class_map"])

#     patch_dir = os.path.join(config.model_dir, "patches", str(uuid.uuid4()))
#     os.makedirs(patch_dir)
#     annotated_patches_record_path = os.path.join(patch_dir, "annotated-patches-record.tfrec")
#     annotated_patches_with_boxes_record_path = os.path.join(patch_dir, "annotated-patches-with-boxes-record.tfrec")
#     annotated_patches_with_no_boxes_record_path = os.path.join(patch_dir, "annotated-patches-with-no-boxes-record.tfrec")
#     unannotated_patches_record_path = os.path.join(patch_dir, "unannotated-patches-record.tfrec")
#     print("writing patches to {} and {}".format(annotated_patches_record_path, unannotated_patches_record_path))
    

#     annotated_tf_records = []
#     annotated_tf_records_with_boxes = []
#     annotated_tf_records_with_no_boxes = []
#     unannotated_tf_records = []

#     for image in tqdm.tqdm(dataset.images, desc="Generating patches"):
        
#         is_annotated = annotations[image.image_name]["status"] == "completed"
#         image_patches = extract_patches_from_image(image, dataset.patch_extraction_params, annotations[image.image_name])
#                                                  #boxes=boxes[image.image_name], classes=classes[image.image_name])

#         write_patches(patch_dir, image_patches)
#         tf_records_for_image = tf_record_io.create_patch_tf_records_for_image(image, image_patches, patch_dir, is_annotated=is_annotated)
        


#         if is_annotated:
#             annotated_tf_records.extend(tf_records_for_image)


#             tf_records_for_image_with_boxes = tf_record_io.create_patch_tf_records_for_image(image, image_patches, 
#                                                                                              patch_dir, is_annotated, 
#                                                                                              require_box=True)

#             tf_records_for_image_with_no_boxes = tf_record_io.create_patch_tf_records_for_image(image, image_patches, 
#                                                                                              patch_dir, is_annotated, 
#                                                                                              require_no_box=True)

#             annotated_tf_records_with_boxes.extend(tf_records_for_image_with_boxes) 
#             annotated_tf_records_with_no_boxes.extend(tf_records_for_image_with_no_boxes)  


#         else:
#             unannotated_tf_records.extend(tf_records_for_image)


#     tf_record_io.output_patch_tf_records(annotated_patches_record_path, annotated_tf_records)
#     tf_record_io.output_patch_tf_records(annotated_patches_with_boxes_record_path, annotated_tf_records_with_boxes)
#     tf_record_io.output_patch_tf_records(annotated_patches_with_no_boxes_record_path, annotated_tf_records_with_no_boxes)
#     tf_record_io.output_patch_tf_records(unannotated_patches_record_path, unannotated_tf_records)


#     return patch_dir


# def extract_patches_for_graph_match(dataset, method): #use_full_patches):
#     annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})
#     completed_images = get_completed_images(annotations)
#     patches = []

#     if len(completed_images) == 0:
#         return patches
#     else:
#         if method != "box_patches":
#             patch_size = get_patch_size(annotations)

#         for image in tqdm.tqdm(dataset.images, desc="Generating patches"):
            
#             #image_patches = extract_patches_from_image(image, dataset.patch_extraction_params, annotations[image.image_name]) #boxes=None, classes=None) #boxes[image.image_name], classes=classes[image.image_name])
#             if annotations[image.image_name]["status"] == "completed":
#                 if method == "patch_surrounding_boxes":
#                     image_patches = extract_patches_surrounding_gt_boxes(image, annotations, patch_size)
#                 elif method == "excess_green_patches":
#                     image_patches = extract_patches_excess_green(image, annotations, patch_size)
#                 elif method == "box_patches":
#                     image_patches = extract_gt_boxes(image, annotations[image.image_name]["boxes"])


#                 patches.extend(image_patches)

#         return patches

# def extract_patches_for_graph(dataset, patch_dir):

#     if os.path.exists(patch_dir):
#        shutil.rmtree(patch_dir)
#     os.makedirs(patch_dir) 

#     annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})

#     #patch_dir = os.path.join(config.model_dir, "patches", str(uuid.uuid4()))
#     #os.makedirs(patch_dir)
#     #patches_record_path = os.path.join(patch_dir, "patches-record.tfrec")
#     #print("writing patches to ", patches_record_path)
    

#     tf_records = []

#     for image in tqdm.tqdm(dataset.images, desc="Generating patches"):
        
#         #image_patches = extract_patches_from_image(image, dataset.patch_extraction_params, annotations[image.image_name]) #boxes=None, classes=None) #boxes[image.image_name], classes=classes[image.image_name])
#         if annotations[image.image_name]["status"] == "completed":
#             image_patches = extract_gt_boxes(image, annotations[image.image_name]["boxes"])

#             write_patches(patch_dir, image_patches)
#         #tf_records_for_image = tf_record_io.create_patch_tf_records_for_image(image, image_patches, patch_dir, is_annotated=True)
#         #tf_records.extend(tf_records_for_image)

#             tf_records_for_image = tf_record_io.create_patch_tf_records_for_image(image, image_patches, patch_dir, is_annotated=False)
#             tf_records.extend(tf_records_for_image)

#     patches_record_path = os.path.join(patch_dir, "patches-record.tfrec")
#     tf_record_io.output_patch_tf_records(patches_record_path, tf_records)


#     #return patch_dir




# def get_patch_size_for_gt_box(gt_box):
#     gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
#     patch_area = gt_box_area * (90000 / 2500)
#     patch_size = round(m.sqrt(patch_area))
#     return patch_size



def write_patches_from_gt_box_records(gt_patch_records, patch_dir):
    extraction_record = {}

    for gt_patch_record in gt_patch_records:
        image_path = gt_patch_record["image_path"]
        gt_coords = gt_patch_record["patch_coords"]

        annotations_path = os.path.join("/".join(image_path.split("/")[:-2]), "annotations", "annotations_w3c.json")
        if annotations_path not in extraction_record:

            extraction_record[annotations_path] = {}
        if image_path not in extraction_record[annotations_path]:
            extraction_record[annotations_path][image_path] = []
        extraction_record[annotations_path][image_path].append(gt_coords)

    patch_num = 0
    patch_records = []
    for annotations_path in extraction_record.keys():
        print("processing", annotations_path)
        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

        patch_size = w3c_io.get_patch_size(annotations)
        for image_path in extraction_record[annotations_path].keys():
            image = Image(image_path)
            gt_boxes = annotations[image.image_name]["boxes"]
            gt_classes = annotations[image.image_name]["classes"]
            image_array = image.load_image_array()

            for gt_box in extraction_record[annotations_path][image_path]:
                patch, patch_coords = _extract_patch_surrounding_gt_box(image_array, gt_box, patch_size)
                patch_record = {}
                patch_record["patch"] = patch
                patch_record["patch_coords"] = patch_coords
                patch_record["image_path"] = image_path
                patch_record["image_name"] = image.image_name
                patch_record["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
                annotate_patch(patch_record, gt_boxes, gt_classes)
                
                patch_records.append(patch_record)
                patch_num += 1


    write_annotated_patch_records(patch_records, patch_dir)



def write_annotated_patch_records(patch_records, patch_dir):

    write_patches(patch_dir, patch_records)
    annotated_tf_records = tf_record_io.create_patch_tf_records(patch_records, patch_dir, is_annotated=True)
    annotated_patches_record_path = os.path.join(patch_dir, "annotated-patches-record.tfrec")
    tf_record_io.output_patch_tf_records(annotated_patches_record_path, annotated_tf_records)






def _extract_patch_surrounding_gt_box(image_array, gt_box, patch_size):

    logger = logging.getLogger(__name__)

    img_h, img_w = image_array.shape[:2]

    box_y_min, box_x_min, box_y_max, box_x_max = gt_box

    box_h = box_y_max - box_y_min
    box_w = box_x_max - box_x_min

    if box_h > patch_size or box_w > patch_size:
        logger.warning("Box exceeds size of patch. (box_w, box_h): ({}, {}). patch_size: {}.".format(
            box_w, box_h, patch_size
        ))
        #raise RuntimeError("Box exceeds size of patch. (box_w, box_h): ({}, {}). patch_size: {}.".format(
        #    box_w, box_h, patch_size
        #))

    patch_y_min = random.randrange((box_y_min + box_h) - max(box_h, patch_size), box_y_min + 1)
    patch_y_min = min(img_h - patch_size, max(0, patch_y_min))
    patch_x_min = random.randrange((box_x_min + box_w) - max(box_w, patch_size), box_x_min + 1)
    patch_x_min = min(img_w - patch_size, max(0, patch_x_min))
    patch_y_max = patch_y_min + patch_size
    patch_x_max = patch_x_min + patch_size

    patch_coords = [patch_y_min, patch_x_min, patch_y_max, patch_x_max]

    if patch_y_min < 0 or patch_x_min < 0 or patch_y_max > img_h or patch_x_max > img_w:
        raise RuntimeError("Patch exceeds boundaries of the image.")

    patch = image_array[patch_y_min:patch_y_max, patch_x_min:patch_x_max]

    return patch, patch_coords



def extract_plant_and_other(image, image_annotations, num_plant, num_other, patch_size, allow_box_reuse=True):

    patch_records = []
    image_array = image.load_image_array()
    gt_boxes = image_annotations["boxes"]
    gt_classes = image_annotations["classes"]
    num_boxes = gt_boxes.shape[0]


    if num_plant == "all":
        num_plant = num_boxes

    if num_plant > num_boxes and not allow_box_reuse:
        raise RuntimeError("Insufficient number of boxes to satisfy request")


    subset_inds = []
    num_needed = num_plant
    while num_needed > 0:
        num_taken = min(num_needed, num_boxes)
        subset_inds.extend(np.random.choice(np.arange(num_boxes), num_taken, replace=False))
        num_needed -= num_taken

    gt_boxes_subset = gt_boxes[subset_inds]

    # if num_plant <= gt_boxes.shape[0] or allow_box_reuse:
    #     if gt_boxes.shape[0] > 0:
    #         subset_inds = np.random.choice(np.arange(gt_boxes.shape[0]), num_plant)
    #         gt_boxes_subset = gt_boxes[subset_inds]
    #     else:
    #         gt_boxes_subset = np.array([])
    #else:
    #    raise RuntimeError("Insufficient number of boxes to satisfy request")

    patch_num = 0
    for gt_box in gt_boxes_subset:
        patch, patch_coords = _extract_patch_surrounding_gt_box(image_array, gt_box, patch_size)
        patch_record = {}
        patch_record["patch"] = patch
        patch_record["patch_coords"] = patch_coords
        patch_record["image_path"] = image.image_path
        patch_record["image_name"] = image.image_name
        patch_record["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
        annotate_patch(patch_record, gt_boxes, gt_classes)
        patch_records.append(patch_record)
        patch_num += 1    


    image_array = np.float32(image_array) / 255
    exg_array = (2 * image_array[:,:,1]) - image_array[:,:,0] - image_array[:,:,2]
    exg_array = image_utils.scale_image(exg_array, -2, 2, 0, 1)

    img_h, img_w = image_array.shape[:2]
    # mask out gt boxes
    pad = 10
    for gt_box in gt_boxes:
        exg_array[max(0, gt_box[0] - pad):min(img_h, gt_box[2] + pad), 
                  max(0, gt_box[1] - pad):min(img_w, gt_box[3] + pad)] = 0.5

    coverage = 4
    #patch_coords_lst = generate_random_patch_coords_lst(image, coverage * num_other, patch_size)
    image_area = img_w * img_h
    patch_area = patch_size * patch_size

    pool_size = max(coverage * num_other, m.ceil(image_area / patch_area))
    patch_coords_lst = generate_evenly_sampled_patch_coords_lst(image, pool_size, patch_size)
    #patch_coords_lst = generate_evenly_sampled_patch_coords_lst(image, coverage * num_other, patch_size)
    exg_patch_records = extract_patch_records_from_image(image, patch_coords_lst, image_annotations, starting_patch_num=num_other)
    exg_patches = extract_patches_from_image_array(exg_array, patch_coords_lst)

    ranks = []
    for exg_patch in exg_patches:
        ranks.append((-1) * (np.sum(exg_patch)))
    #inds = np.argsort(np.array(ranks))
    #exg_patch_records = (np.array(exg_patch_records)[inds][:num_other]).tolist()

    inds = np.arange(len(ranks))
    epsilon = 1e-10
    probs = ranks / (np.sum(ranks) + epsilon)
    sel_inds = np.random.choice(inds, num_other, p=probs, replace=False)
    exg_patch_records = (np.array(exg_patch_records)[sel_inds]).tolist()

    return patch_records, exg_patch_records


def extract_patch_records_with_exg_box_combo(image, image_annotations, num_patches, patch_size):

    prop_box_patches = 0.80
    tentative_num_box_patches = m.ceil(prop_box_patches * num_patches)


    patch_records = []
    image_array = image.load_image_array()
    gt_boxes = image_annotations["boxes"]
    gt_classes = image_annotations["classes"]

    if tentative_num_box_patches < gt_boxes.shape[0]:
        subset_inds = np.random.choice(np.arange(gt_boxes.shape[0]), tentative_num_box_patches, replace=False)
        gt_boxes_subset = gt_boxes[subset_inds]
    else:
        gt_boxes_subset = gt_boxes

    patch_num = 0
    for gt_box in gt_boxes_subset:
        patch, patch_coords = _extract_patch_surrounding_gt_box(image_array, gt_box, patch_size)
        patch_record = {}
        patch_record["patch"] = patch
        patch_record["patch_coords"] = patch_coords
        patch_record["image_path"] = image.image_path
        patch_record["image_name"] = image.image_name
        patch_record["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
        annotate_patch(patch_record, gt_boxes, gt_classes)
        patch_records.append(patch_record)
        patch_num += 1


    num_box_patches = gt_boxes_subset.shape[0]
    num_exg_patches = num_patches - num_box_patches

    image_array = np.float32(image_array) / 255
    exg_array = (2 * image_array[:,:,1]) - image_array[:,:,0] - image_array[:,:,2]
    exg_array = image_utils.scale_image(exg_array, -2, 2, 0, 1)

    img_h, img_w = image_array.shape[:2]
    # mask out gt boxes
    pad = 10
    for gt_box in gt_boxes:
        exg_array[max(0, gt_box[0] - pad):min(img_h, gt_box[2] + pad), 
                  max(0, gt_box[1] - pad):min(img_w, gt_box[3] + pad)] = 0.5

    coverage = 4
    #patch_coords_lst = generate_random_patch_coords_lst(image, coverage * num_exg_patches, patch_size)
    image_area = img_w * img_h
    patch_area = patch_size * patch_size

    pool_size = max(coverage * num_patches, m.ceil(image_area / patch_area))
    patch_coords_lst = generate_evenly_sampled_patch_coords_lst(image, pool_size, patch_size)
    #patch_coords_lst = generate_evenly_sampled_patch_coords_lst(image, coverage * num_exg_patches, patch_size)
    exg_patch_records = extract_patch_records_from_image(image, patch_coords_lst, image_annotations, starting_patch_num=num_box_patches)
    exg_patches = extract_patches_from_image_array(exg_array, patch_coords_lst)

    ranks = []
    for exg_patch in exg_patches:
        ranks.append((-1) * (np.sum(exg_patch)))

    inds = np.arange(len(ranks))
    epsilon = 1e-10
    probs = ranks / (np.sum(ranks) + epsilon)
    sel_inds = np.random.choice(inds, num_exg_patches, p=probs, replace=False)
    #inds = np.argsort(np.array(ranks))
    #exg_patch_records = (np.array(exg_patch_records)[inds][:num_exg_patches]).tolist()
    exg_patch_records = (np.array(exg_patch_records)[sel_inds]).tolist()
    patch_records.extend(exg_patch_records)

    #del image_array
    #del exg_array

    return patch_records


def extract_patch_records_randomly(image, image_annotations, num_patches, patch_size):

    patch_coords_lst = generate_evenly_sampled_patch_coords_lst(image, num_patches, patch_size)
    patch_records = extract_patch_records_from_image(image, patch_coords_lst, image_annotations)
    return patch_records


def extract_patch_records_with_exg(image, image_annotations, num_patches, patch_size):

    # TODO: change to a random extraction -- just randomly e.g., extract (4 * num_patches) patches from the image
    # and then pick the (num_patches) most green ones
    w, h = image.get_wh()
    image_area = w * h
    patch_area = patch_size * patch_size
    coverage = 4
    pool_size = max(coverage * num_patches, m.ceil(image_area / patch_area))
    patch_coords_lst = generate_evenly_sampled_patch_coords_lst(image, pool_size, patch_size)
    #patch_coords_lst = generate_random_patch_coords_lst(image, 4 * num_patches, patch_size)
    patch_records = extract_patch_records_from_image(image, patch_coords_lst, image_annotations)
    #patches = extract_patches_from_image_tiled(image, patch_size, annotations, patch_overlap_percent=85)

    image_array = np.float32(image.load_image_array()) / 255
    exg_array = (2 * image_array[:,:,1]) - image_array[:,:,0] - image_array[:,:,2]
    exg_array = image_utils.scale_image(exg_array, -2, 2, 0, 1)
    exg_patches = extract_patches_from_image_array(exg_array, patch_coords_lst)
    #extract_patches_from_image_array_tiled(excess_green, patch_size, patch_overlap_percent=85)

    ranks = []
    for exg_patch in exg_patches:
        #ranks.append((-1) * (np.sum(exg_patch[exg_patch > np.percentile(exg_patch, 85)])))
        #ranks.append((-1) * (np.std(exg_patch)))
        ranks.append((-1) * (np.sum(exg_patch))) # np.sum(np.interp(exg_patch, (-1, 1), (0, 1)))))
    #inds = np.argsort(np.array(ranks))
    #patch_records = (np.array(patch_records)[inds][:num_patches]).tolist()
    
    inds = np.arange(len(ranks))
    epsilon = 1e-10
    probs = ranks / (np.sum(ranks) + epsilon)
    sel_inds = np.random.choice(inds, num_patches, p=probs, replace=False)
    patch_records = (np.array(patch_records)[sel_inds]).tolist()    
    
    return patch_records

def extract_patch_records_surrounding_gt_boxes(image, image_annotations, num_patches, patch_size):

    patch_records = []
    image_array = image.load_image_array()
    gt_boxes = image_annotations["boxes"]
    gt_classes = image_annotations["classes"]
    num_boxes = gt_boxes.shape[0]
    patch_num = 0

    subset_inds = []
    num_needed = num_patches
    while num_needed > 0:
        num_taken = min(num_needed, num_boxes)
        subset_inds.extend(np.random.choice(np.arange(num_boxes), num_taken, replace=False).tolist())
        num_needed -= num_taken
    #gt_boxes_subset = gt_boxes[subset_inds]
    #gt_classes_subset = gt_classes[subset_inds]

    #num_patches_needed = num_patches
    #for gt_box in gt_boxes:
    #while num_patches_needed > 0:
    for subset_ind in subset_inds:
        gt_box = gt_boxes[subset_ind] #patch_num % num_boxes]
        patch, patch_coords = _extract_patch_surrounding_gt_box(image_array, gt_box, patch_size)
        patch_record = {}
        patch_record["patch"] = patch
        patch_record["patch_coords"] = patch_coords
        patch_record["image_path"] = image.image_path
        patch_record["image_name"] = image.image_name
        patch_record["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
        annotate_patch(patch_record, gt_boxes, gt_classes)
        patch_records.append(patch_record)
        patch_num += 1
        #num_patches_needed -= 1

    return patch_records



def extract_gt_boxes(image, gt_boxes):

    image_array = image.load_image_array()

    #print("image_array shape", np.shape(image_array))

    patch_num = 0
    gt_patches = []
    max_h = 0
    max_w = 0
    for gt_box in gt_boxes:
        gt_patch = image_array[gt_box[0]:gt_box[2], gt_box[1]:gt_box[3], :]
        #print(np.shape(gt_patch), end=" ")
        h = gt_box[2] - gt_box[0]
        w = gt_box[3] - gt_box[1]
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

        if h > 1 and w > 1:
            patch_data = {}
            patch_data["patch"] = gt_patch
            patch_data["image_name"] = image.image_name
            patch_data["image_path"] = image.image_path
            patch_data["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
            patch_data["patch_coords"] = gt_box
            gt_patches.append(patch_data)
            patch_num += 1

    #print("max_h", max_h)
    #print("max_w", max_w)

    return gt_patches



def write_patches(out_dir, patch_data_lst):
    for patch_data in patch_data_lst:
        #print(np.shape(patch_data["patch"]), end=" ")
        #h, w = np.shape(patch_data["patch"])[:2]
        #if h > 1 or w > 1:
        cv2.imwrite(os.path.join(out_dir, patch_data["patch_name"]),
                cv2.cvtColor(patch_data["patch"], cv2.COLOR_RGB2BGR))


def extract_patch_around_box(image, box, patch_size):
    logger = logging.getLogger(__name__)

    image_array = image.load_image_array()
    img_h, img_w = image_array.shape[:2]


    box_y_min, box_x_min, box_y_max, box_x_max = box

    box_h = box_y_max - box_y_min
    box_w = box_x_max - box_x_min

    #if box_h > patch_size or box_w > patch_size:
    #    raise RuntimeError("Box exceeds size of patch.")

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

    patch = image_array[patch_y_min:patch_y_max, patch_x_min:patch_x_max]

    return patch, patch_coords


# def extract_patches_from_image(image, patch_extraction_params, image_annotations):

#     method = patch_extraction_params["method"]
#     if method == "tile":
#         image_patches = extract_patches_from_image_tile(image, patch_extraction_params, image_annotations)
#     else:
#         raise RuntimeError("Unsupported patch extaction method")
    
#     return image_patches


def extract_patch_records_from_image_tiled(image, patch_size, image_annotations, patch_overlap_percent=50, starting_patch_num=0):

    annotation_status = image_annotations["status"]
    annotation_boxes = image_annotations["boxes"]
    annotation_classes = image_annotations["classes"]

    image_patches = []

    image_array = image.load_image_array()

    tile_size = patch_size
    overlap_px = int(m.floor(tile_size * (patch_overlap_percent / 100)))

    h, w = image_array.shape[:2]
    patch_num = starting_patch_num
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
            patch_data["image_name"] = image.image_name
            patch_data["image_path"] = image.image_path
            patch_data["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
            patch_data["patch_coords"] = patch_coords
            
            if annotation_status == "completed":
                annotate_patch(patch_data, annotation_boxes, annotation_classes)
            image_patches.append(patch_data)
            patch_num += 1

    return image_patches


def extract_patches_from_image_array_tiled(image_array, patch_size, patch_overlap_percent=50):

    tile_size = patch_size
    overlap_px = int(m.floor(tile_size * (patch_overlap_percent / 100)))

    h, w = image_array.shape[:2]
    patches = []
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
            patches.append(patch)

    return patches  


# def _extract_random_patch(image_array, patch_size):

#     img_h, img_w = image_array.shape[:2]

#     patch_y_min = random.randrange(0, img_h - patch_size)
#     patch_x_min = random.randrange(0, img_w - patch_size)
#     patch_y_max = patch_y_min + patch_size
#     patch_x_max = patch_x_min + patch_size

#     patch_coords = [patch_y_min, patch_x_min, patch_y_max, patch_x_max]

#     patch = image_array[patch_y_min:patch_y_max, patch_x_min:patch_x_max]

#     return patch, patch_coords


def get_factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def generate_evenly_sampled_patch_coords_lst(image, num_patches, patch_size):

    img_w, img_h = image.get_wh()
    factors = sorted(get_factors(num_patches))

    image_ratio = max(img_w, img_h) / min(img_w, img_h)

    best_factors = (num_patches, 1)
    best_ratio_diff = np.inf
    i = 0
    #for i in range((len(factors) // 2) + 1):
    while (factors[-i-1] >= factors[i]):
        factor_ratio = factors[-i-1] / factors[i]
        ratio_diff = abs(image_ratio - factor_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_factors = (factors[-i-1], factors[i])
        i += 1

    if img_w > img_h:
        #width_step = img_w // best_factors[0]
        #height_step = img_h // best_factors[1]
        #x_positions = np.rint(np.linspace(0, img_w, best_factors[0]+1, endpoint=True))
        #y_positions = np.rint(np.linspace(0, img_h, best_factors[1]+1, endpoint=True))
        x_positions = np.rint(np.linspace(patch_size // 2, img_w - patch_size // 2, best_factors[0]+1, endpoint=True))
        y_positions = np.rint(np.linspace(patch_size // 2, img_h - patch_size // 2, best_factors[1]+1, endpoint=True))
    else:
        #width_step = img_w // best_factors[1]
        #height_step = img_h // best_factors[0]
        #x_positions = np.rint(np.linspace(0, img_w, best_factors[1]+1, endpoint=True))
        #y_positions = np.rint(np.linspace(0, img_h, best_factors[0]+1, endpoint=True))
        x_positions = np.rint(np.linspace(patch_size // 2, img_w - patch_size // 2, best_factors[1]+1, endpoint=True))
        y_positions = np.rint(np.linspace(patch_size // 2, img_h - patch_size // 2, best_factors[0]+1, endpoint=True))


    #print("x_positions: {}, y_positions: {}, patch_size: {}, img_w: {}, img_h: {}".format(
    #    x_positions, y_positions, patch_size, img_w, img_h
    #))

    patch_coords_lst = []
    #for i in range(0, img_w, width_step):
    #    for j in range(0, img_h, height_step):

    for i in range(len(x_positions)-1):
        for j in range(len(y_positions)-1):
            #patch_y_centre = random.randrange(j, j+height_step)
            patch_y_centre = random.randrange(y_positions[j], y_positions[j+1]+1)
            patch_y_min = patch_y_centre - (patch_size // 2)
            patch_y_min = min(img_h - patch_size, max(0, patch_y_min))

            #patch_x_centre = random.randrange(i, i+width_step)
            patch_x_centre = random.randrange(x_positions[i], x_positions[i+1]+1)
            patch_x_min = patch_x_centre - (patch_size // 2)
            patch_x_min = min(img_w - patch_size, max(0, patch_x_min))

            #patch_y_min = random.randrange(min(j, img_h - patch_size), min(j+height_step, img_h - patch_size + 1))

            #patch_y_min = random.randrange(0, j + height_step - patch_size)
            #patch_x_min = random.randrange(0, i + width_step - patch_size)
            patch_y_max = patch_y_min + patch_size
            patch_x_max = patch_x_min + patch_size

            patch_coords_lst.append([patch_y_min, patch_x_min, patch_y_max, patch_x_max])
    

    assert(len(patch_coords_lst) == num_patches)

    return patch_coords_lst




def generate_random_patch_coords_lst(image, num_patches, patch_size):

    img_w, img_h = image.get_wh()

    patch_coords_lst = []
    for _ in range(num_patches):

        patch_y_min = random.randrange(0, img_h - patch_size)
        patch_x_min = random.randrange(0, img_w - patch_size)
        patch_y_max = patch_y_min + patch_size
        patch_x_max = patch_x_min + patch_size

        patch_coords_lst.append([patch_y_min, patch_x_min, patch_y_max, patch_x_max])

    return patch_coords_lst

def extract_patch_records_from_image(image, patch_coords_lst, image_annotations, starting_patch_num=0):

    annotation_status = image_annotations["status"]
    annotation_boxes = image_annotations["boxes"]
    annotation_classes = image_annotations["classes"]
    
    image_patches = []
    image_array = image.load_image_array()

    patch_num = starting_patch_num
    for patch_coords in patch_coords_lst:
        patch = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
        #patch, patch_coords = _extract_random_patch(image_array, patch_size)

        patch_data = {}
        patch_data["patch"] = patch
        patch_data["image_name"] = image.image_name
        patch_data["image_path"] = image.image_path
        patch_data["patch_name"] = image.image_name + "-patch-" + str(patch_num).zfill(5) + ".png"
        patch_data["patch_coords"] = patch_coords
        
        if annotation_status == "completed":
            annotate_patch(patch_data, annotation_boxes, annotation_classes)
        image_patches.append(patch_data)
        patch_num += 1

    return image_patches

def extract_patches_from_image_array(image_array, patch_coords_lst):
    
    image_patches = []
    for patch_coords in patch_coords_lst:
        patch = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
        image_patches.append(patch)

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

        patch = patch_data["patch"]
        patch_coords = patch_data["patch_coords"] 

        centres = np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)
        contained_inds = get_contained_inds(centres, patch_data["patch_coords"])
        contained_boxes = gt_boxes[contained_inds]
        contained_classes = gt_classes[contained_inds]

        #patch_height = patch_coords[2] - patch_coords[0]
        #patch_width = patch_coords[3] - patch_coords[1]
        image_abs_boxes, mask = box_utils.clip_boxes_and_get_small_visibility_mask(
            contained_boxes, patch_coords, min_visibility=0.05)

        image_abs_boxes = image_abs_boxes[mask]
        contained_classes = contained_classes[mask]

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