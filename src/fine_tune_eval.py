import logging
import os
import glob
import shutil
import tqdm
import uuid
import urllib3
import time
import random
import requests
import json
import math as m
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from natsort import natsorted

from mean_average_precision import MetricBuilder
# import traceback
# import numpy as np
# import glob

# import image_set_aux
# import image_set_actions as isa
# import extract_patches as ep
# from image_set import Image
# import excess_green

# from io_utils import json_io, w3c_io, tf_record_io
# from models.yolov4 import yolov4_image_set_driver
# from models.common import annotation_utils, inference_metrics
# import auto_select
from io_utils import json_io
from models.common import annotation_utils, box_utils, inference_metrics
# from fine_tune_dist_select import get_selected_order_for_image_set

# from server import process_train, process_predict

import server
from lock_queue import LockQueue


sch_ctx = {}




server_url = "http://" + os.environ.get("CC_IP") + ":" + os.environ.get("CC_PY_PORT") + os.environ.get("CC_PATH")

def region_is_valid(new_region, annotations, image_name, image_h, image_w):
    if new_region[0] < 0 or new_region[1] < 0 or new_region[2] > image_h or new_region[3] > image_w:
        return False

    for training_region in annotations[image_name]["training_regions"]:
        if box_utils.get_intersection_rect(new_region, training_region)[0]:
            return False
    for test_region in annotations[image_name]["test_regions"]:
        if box_utils.get_intersection_rect(new_region, test_region)[0]:
            return False

    return True



def get_valid_child_regions(reg, annotations, image_name, image_h, image_w):
    child_regions = []
    REGION_GROWTH_STEP_SIZE = 10
    for direction_index in range(4):
        if direction_index == 0 or direction_index == 1:
            sign = -1
        else:
            sign = 1
        child_region = np.copy(reg)
        child_region[direction_index] = child_region[direction_index] + (sign) * REGION_GROWTH_STEP_SIZE
        if region_is_valid(child_region, annotations, image_name, image_h, image_w):
            child_regions.append(child_region.tolist())

    random.shuffle(child_regions)
    return child_regions



    # reg_0 = [reg[0] - REGION_GROWTH_STEP_SIZE, reg[1], reg[2], reg[3]]
    # if region_is_valid(reg_0, annotations, image_name, image_h, image_w):
    #     child_regions.append(reg_0)
    # reg_1 = [reg[0], reg[1] - REGION_GROWTH_STEP_SIZE, reg[2], reg[3]]
    # if region_is_valid(reg_0, annotations, image_name, image_h, image_w):
    #     child_regions.append(reg_1)


def confidence_guided_region(annotations, predictions, min_num_contained_objects, image_name, image_h, image_w):

    NUM_NEIGHBOURS = 10
    STEP_SIZE = 100


    all_regions = []
    all_regions.extend(annotations[image_name]["training_regions"])
    all_regions.extend(annotations[image_name]["test_regions"])

    candidate_pts = []
    for y in range(int(STEP_SIZE/2), image_h, STEP_SIZE):
        for x in range(int(STEP_SIZE/2), image_w, STEP_SIZE):
            
            candidate_pts.append([y, x])

    candidate_pts = np.array(candidate_pts)
    inds = box_utils.get_contained_inds_for_points(candidate_pts, all_regions)
    mask = np.full(candidate_pts.shape[0], True)
    mask[inds] = False
    candidate_pts = candidate_pts[mask]

    # boxes = np.array(annotations[image_name]["boxes"])
    # inds = box_utils.get_contained_inds(boxes, all_regions)

    # mask = np.full(boxes.shape[0], True)
    # mask[inds] = False

    # uncontained_boxes = boxes[mask]



    pred_boxes = np.array(predictions[image_name]["boxes"])
    pred_scores = np.array(predictions[image_name]["scores"])
    inds = box_utils.get_contained_inds(pred_boxes, all_regions)

    mask = np.full(pred_boxes.shape[0], True)
    mask[inds] = False

    uncontained_pred_boxes = pred_boxes[mask]
    uncontained_pred_scores = pred_scores[mask]
    # uncontained_boxes = annotations[image_name]["boxes"][mask]

    pred_centres = (uncontained_pred_boxes[..., :2] + uncontained_pred_boxes[..., 2:]) / 2.0

    # if centres.shape[0] < min_num_contained_objects:
    #     return None
    # print("centres.shape", centres.shape)




    point_qualities = []
    if pred_centres.shape[0] == 0:
        return random_region(annotations, min_num_contained_objects, image_name, image_h, image_w)

    tree = KDTree(pred_centres)
    k = min(NUM_NEIGHBOURS, pred_centres.shape[0])
    for candidate_pt in tqdm.tqdm(candidate_pts):
        dist, ind = tree.query(candidate_pt.reshape(1, -1), k=k)

        neighbour_scores = uncontained_pred_scores[ind]
        point_quality = get_confidence_quality(neighbour_scores)
        point_qualities.append(point_quality)

    sorted_quality_inds = np.argsort(point_qualities)
    best_pt = candidate_pts[sorted_quality_inds[0]]

    # print(best_pt)

    return search_vertices(best_pt, annotations, min_num_contained_objects, image_name, image_h, image_w) #search_parent(best_pt, annotations, min_num_contained_objects, image_name, image_h, image_w)



def search_vertices(start_pt, annotations, min_num_contained_objects, image_name, image_h, image_w):


    all_ys = [0, image_h]
    all_xs = [0, image_w]
    for training_region in annotations[image_name]["training_regions"]:
        all_ys.extend([training_region[0], training_region[2]])
        all_xs.extend([training_region[1], training_region[3]])
    for test_region in annotations[image_name]["test_regions"]:
        all_ys.extend([test_region[0], test_region[2]])
        all_xs.extend([test_region[1], test_region[3]])

    vertices = []
    for y in all_ys:
        for x in all_xs:
            vertices.append([y, x])

    region_vertices = []
    for training_region in annotations[image_name]["training_regions"]:
        region_vertices.append([training_region[0], training_region[1]])
        region_vertices.append([training_region[0], training_region[3]])
        region_vertices.append([training_region[2], training_region[1]])
        region_vertices.append([training_region[2], training_region[3]])

    for test_region in annotations[image_name]["test_regions"]:
        region_vertices.append([test_region[0], test_region[1]])
        region_vertices.append([test_region[0], test_region[3]])
        region_vertices.append([test_region[2], test_region[1]])
        region_vertices.append([test_region[2], test_region[3]])

    region_vertices = np.array(region_vertices)
    # print("region_vertices", region_vertices)


    # vertices = []
    # for training_region in annotations[image_name]["training_regions"]:
    #     vertices.append([training_region[0], training_region[1]])
    #     vertices.append([training_region[2], training_region[3]])
    # for test_region in annotations[image_name]["test_regions"]:
    #     vertices.append([test_region[0], test_region[1]])
    #     vertices.append([test_region[2], test_region[3]])    
    # vertices.append([0, 0])
    # vertices.append([image_h, image_w])
    
    start_pt = np.array(start_pt)
    vertices = np.array(vertices)

    dists = np.linalg.norm(start_pt - vertices, axis=1)
    inds = np.argsort(dists)
    sorted_vertices = vertices[inds]

    for v1 in sorted_vertices:
        for v2 in sorted_vertices:
            
            if v1[0] == v2[0] or v1[1] == v2[1]:
                continue
            reg = np.array([
                min(v1[0], v2[0]),
                min(v1[1], v2[1]),
                max(v1[0], v2[0]),
                max(v1[1], v2[1])    
            ])
            # print("processing region", reg)

            c_inds = box_utils.get_contained_inds_for_points(np.array([start_pt]), [reg])
            if c_inds.size == 0:
                # print("does not contain start_pt")
                continue
        
            # c_inds = box_utils.get_contained_inds_for_points(region_vertices, [reg])
            intersects = False
            for training_region in annotations[image_name]["training_regions"]:
                if box_utils.get_intersection_rect(reg, training_region)[0]:
                    intersects = True
                    continue
            for test_region in annotations[image_name]["test_regions"]:
                if box_utils.get_intersection_rect(reg, test_region)[0]:
                    intersects = True
                    continue

            if intersects:
                # print("intersects")
                continue

            # print("num_contained_vertices", c_inds.size)
            # print("sorted_vertices[c_inds]", sorted_vertices[c_inds])
            # if c_inds.size > 0:
            #     continue

            c_inds = box_utils.get_fully_contained_inds(annotations[image_name]["boxes"], [reg])
            # print("num_contained_boxes", c_inds.size)
            if c_inds.size >= min_num_contained_objects:
                # return True
                contained_boxes = annotations[image_name]["boxes"][c_inds]
                contained_box_centres = (contained_boxes[..., :2] + contained_boxes[..., 2:]) / 2.0
                sorted_inds = np.argsort(np.linalg.norm(start_pt-contained_box_centres, axis=1))
                chosen_boxes = contained_boxes[sorted_inds[:min_num_contained_objects]]
                chosen_boxes = np.concatenate([chosen_boxes, np.expand_dims(np.tile(start_pt, 2), axis=0)])

                final_region = [
                    int(np.min(chosen_boxes[:,0])),
                    int(np.min(chosen_boxes[:,1])),
                    int(np.max(chosen_boxes[:,2])),
                    int(np.max(chosen_boxes[:,3]))
                ]

                
                # for i in range(4):
                #     if i == 0 or i == 1:
                #         sign = -1
                #     else:
                #         sign = 1
                #     save_coord = final_region[i]
                #     pad = random.randint(10, 20)
                #     final_region[i] = final_region[i] + (sign) * pad
                #     apply_pad = True
                #     for training_region in annotations[image_name]["training_regions"]:
                #         if box_utils.get_intersection_rect(final_region, training_region)[0]:
                #             apply_pad = False
                #     for test_region in annotations[image_name]["test_regions"]:
                #         if box_utils.get_intersection_rect(final_region, test_region)[0]:
                #             apply_pad = False
                #     if i == 0 and final_region[i] < 0:
                #         apply_pad = False
                #     if i == 1 and final_region[i] < 0:
                #         apply_pad = False
                #     if i == 2 and final_region[i] > image_h:
                #         apply_pad = False
                #     if i == 3 and final_region[i] > image_w:
                #         apply_pad = False

                #     if not apply_pad:
                #         final_region[i] = save_coord



                return final_region

    return None

def random_region(annotations, min_num_contained_objects, image_name, image_h, image_w):


    all_regions = []
    all_regions.extend(annotations[image_name]["training_regions"])
    all_regions.extend(annotations[image_name]["test_regions"])

    inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], all_regions)

    mask = np.full(annotations[image_name]["boxes"].shape[0], True)
    mask[inds] = False

    uncontained_boxes = annotations[image_name]["boxes"][mask]

    start_pts = (uncontained_boxes[..., :2] + uncontained_boxes[..., 2:]) / 2.0
    
    # STEP_SIZE = 10
    # start_pts = []
    # for y in range(int(STEP_SIZE/2), image_h, STEP_SIZE):
    #     for x in range(int(STEP_SIZE/2), image_w, STEP_SIZE):
    #         start_pts.append([y, x])

    # random.shuffle(start_pts)
    # start_pts = np.array(start_pts)
    np.random.shuffle(start_pts)
    for i in tqdm.trange(start_pts.shape[0]):
        start_pt = start_pts[i, :]
        # print("start_pt", start_pt)
        # ret = search_parent(start_pt, annotations, min_num_contained_objects, image_name, image_h, image_w)
        ret = search_vertices(start_pt, annotations, min_num_contained_objects, image_name, image_h, image_w)
        if ret is not None:
            return ret

    return None

    
def search_test():
    start_pt = (100, 100) #720) #(10, 10)
    # image_set_dir = os.path.join("usr", "data", "erik", "image_sets", "Blocks2022_reg", "Kernen1_reg", "2022-06-08")
    image_set_dir = os.path.join("usr", "data", "erik", "image_sets", 
    "BlaineLake:sel_img_rand_reg", "Serhienko9S:sel_img_rand_reg", "2022-06-07")
    annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)
    min_num_contained_objects = 10
    image_name = "46"
    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    image_h = metadata["images"][image_name]["height_px"]
    image_w = metadata["images"][image_name]["width_px"]
    # ret = search_parent(start_pt, annotations, min_num_contained_objects, image_name, image_h, image_w)
    ret = search_vertices(start_pt, annotations, min_num_contained_objects, image_name, image_h, image_w)
    print(ret)

def search_parent(start_pt, annotations, min_num_contained_objects, image_name, image_h, image_w):
    seen = {}

    def search(region):

        child_regions = get_valid_child_regions(region, annotations, image_name, image_h, image_w)
        # print("{} {}: {} got {} child regions".format(start_pt[0], start_pt[1], image_name, len(child_regions)))
        for child_region in child_regions:
            str_child_region = str(child_region)
            # print(len(seen.keys()))
            if str_child_region in seen:
                # print("child region seen")
                continue
            seen[str_child_region] = True
            contained_inds = box_utils.get_fully_contained_inds(annotations[image_name]["boxes"], [child_region])
            if contained_inds.size >= min_num_contained_objects:
                return child_region
            else:
                ret = search(child_region)
                if ret is not None:
                    return ret


        return None

    return search([start_pt[0], start_pt[1], start_pt[0], start_pt[1]])



# def create_region_around_point(point_x, point_y, annotations, image_name, min_num_contained_objects):
#     REGION_GROWTH_STEP_SIZE = 10
#     proposed_region = np.array([
#         point_y-1, point_x-1, point_y+1, point_x+1
#     ])
#     seen = {}
#     done = False
#     while not done:
#         # contained_inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], [proposed_region])
#         # if contained_inds.size > min_num_contained_objects:
#         #     break

#         legal_regions = get_legal_regions(proposed_region, annotations)

#         for region in legal_regions:
#             if region in seen:
#                 continue
#             seen[region] = True
#             contained_inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], [region])
#             if contained_inds.size > min_num_contained_objects:
#                 done = True
#                 break

#         if done:
#             break



#         new_min_y = proposed_region[0] - REGION_GROWTH_STEP_SIZE

        


#     return proposed_region


def get_confidence_quality(pred_scores):
    num_predictions = int(pred_scores.size)
    confidence_score = 0
    STEP_SIZE = 0.01
    if num_predictions > 0:
        for conf_val in np.arange(0.25, 1.0, STEP_SIZE): #25, 1.0, STEP_SIZE):
            num_in_range = float(
                np.sum(np.logical_and(pred_scores > conf_val, pred_scores <= conf_val+STEP_SIZE))
            )
            prob = num_in_range / num_predictions
            # confidence_score += prob * (conf_val ** 2) #(2 ** (conf_val* 100)) # * conf_val * conf_val)
            # confidence_score += prob * (conf_val * conf_val)
            confidence_score += prob * (1 / (1 + (m.e ** (-30 * (conf_val - 0.80)))))
            # confidence_score += prob * (2 ** (30 * (conf_val-1))) #( (2**7) * ((conf_val - 0.5) ** 7) )
    return confidence_score



def get_AP(annotations, full_predictions, iou_thresh): #annotated_boxes, predicted_boxes, predicted_scores):
    
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)

    for image_name in annotations.keys():
        annotated_boxes = annotations[image_name]["boxes"]
        predicted_boxes = full_predictions[image_name]["boxes"]
        predicted_scores = full_predictions[image_name]["scores"]

        annotated_classes = np.zeros(shape=(annotated_boxes.shape[0]))
        predicted_classes = np.zeros(shape=(predicted_boxes.shape[0]))

        pred_for_mAP, true_for_mAP = inference_metrics.get_pred_and_true_for_mAP(
            predicted_boxes, 
            predicted_classes, 
            predicted_scores,
            annotated_boxes,
            annotated_classes)

        metric_fn.add(pred_for_mAP, true_for_mAP)

    if iou_thresh == ".50:.05:.95":
        mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".50":
        mAP = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".75":
        mAP = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    else:
        raise RuntimeError("Invalid IoU threshold: {}".format(iou_thresh))

    return mAP

def get_dics(annotations, full_predictions, assessment_images):
    dics = []
    for image_name in assessment_images:
        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(full_predictions[image_name]["boxes"])
        pred_scores = np.array(full_predictions[image_name]["scores"])

        sel_pred_boxes = pred_boxes[pred_scores > 0.50]

        num_predicted = sel_pred_boxes.shape[0]
        num_annotated = annotated_boxes.shape[0]

        dics.append(num_predicted - num_annotated)
    # print(dics)
    return dics

def get_percent_count_errors(annotations, full_predictions, assessment_images):
    errors = []
    for image_name in assessment_images:
        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(full_predictions[image_name]["boxes"])
        pred_scores = np.array(full_predictions[image_name]["scores"])

        sel_pred_boxes = pred_boxes[pred_scores > 0.50]

        num_predicted = sel_pred_boxes.shape[0]
        num_annotated = annotated_boxes.shape[0]

        if num_annotated > 0:

            percent_count_error = abs((num_predicted - num_annotated) / (num_annotated)) * 100

            errors.append(percent_count_error)
    # print(errors)
    return errors

    

def get_global_accuracy(annotations, full_predictions, assessment_images):
    # accuracies = []
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0


    for image_name in assessment_images:
        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(full_predictions[image_name]["boxes"])
        pred_scores = np.array(full_predictions[image_name]["scores"])

        sel_pred_boxes = pred_boxes[pred_scores > 0.50]

        num_predicted = sel_pred_boxes.shape[0]
        num_annotated = annotated_boxes.shape[0]

        if num_predicted > 0:
            if num_annotated > 0:
                true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(annotated_boxes, sel_pred_boxes, 0.50)
                # print(true_positive, false_positive, false_negative)
                # precision_050 = true_positive / (true_positive + false_positive)
                # recall_050 = true_positive / (true_positive + false_negative)
                # if precision_050 == 0 and recall_050 == 0:
                #     f1_iou_050 = 0
                # else:
                #     f1_iou_050 = (2 * precision_050 * recall_050) / (precision_050 + recall_050)
                # acc_050 = true_positive / (true_positive + false_positive + false_negative)
                # true_positive, false_positive, false_negative = get_positives_and_negatives(annotated_boxes, sel_region_pred_boxes, 0.75)
                # precision = true_positive / (true_positive + false_positive)
                # recall = true_positive / (true_positive + false_negative)
                # f1_iou_075 = (2 * precision * recall) / (precision + recall)                        

                
            
            else:
                true_positive = 0
                false_positive = num_predicted
                false_negative = 0

                # precision_050 = 0.0
                # recall_050 = 0.0
                # f1_iou_050 = 0.0
                # acc_050 = 0.0
        else:
            if num_annotated > 0:
                true_positive = 0
                false_positive = 0
                false_negative = num_annotated

                # precision_050 = 0.0
                # recall_050 = 0.0
                # f1_iou_050 = 0.0
                # acc_050 = 0.0
            else:
                true_positive = 0
                false_positive = 0
                false_negative = 0

                # precision_050 = 1.0
                # recall_050 = 1.0
                # f1_iou_050 = 1.0
                # acc_050 = 1.0
        total_true_positives += true_positive
        total_false_positives += false_positive
        total_false_negatives += false_negative


        # accuracies.append(acc_050)
    global_accuracy = total_true_positives / (total_true_positives + total_false_positives + total_false_negatives)
    return global_accuracy

def get_accuracy(annotated_boxes, sel_pred_boxes):

    num_predicted = sel_pred_boxes.shape[0]
    num_annotated = annotated_boxes.shape[0]

    if num_predicted > 0:
        if num_annotated > 0:
            true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(annotated_boxes, sel_pred_boxes, 0.50)
            # print(true_positive, false_positive, false_negative)
            precision_050 = true_positive / (true_positive + false_positive)
            recall_050 = true_positive / (true_positive + false_negative)
            if precision_050 == 0 and recall_050 == 0:
                f1_iou_050 = 0
            else:
                f1_iou_050 = (2 * precision_050 * recall_050) / (precision_050 + recall_050)
            acc_050 = true_positive / (true_positive + false_positive + false_negative)
            # true_positive, false_positive, false_negative = get_positives_and_negatives(annotated_boxes, sel_region_pred_boxes, 0.75)
            # precision = true_positive / (true_positive + false_positive)
            # recall = true_positive / (true_positive + false_negative)
            # f1_iou_075 = (2 * precision * recall) / (precision + recall)                        

            
        
        else:
            true_positive = 0
            false_positive = num_predicted
            false_negative = 0

            precision_050 = 0.0
            recall_050 = 0.0
            f1_iou_050 = 0.0
            acc_050 = 0.0
    else:
        if num_annotated > 0:
            true_positive = 0
            false_positive = 0
            false_negative = num_annotated

            precision_050 = 0.0
            recall_050 = 0.0
            f1_iou_050 = 0.0
            acc_050 = 0.0
        else:
            true_positive = 0
            false_positive = 0
            false_negative = 0

            precision_050 = 1.0
            recall_050 = 1.0
            f1_iou_050 = 1.0
            acc_050 = 1.0

    return acc_050

def get_positives_and_negatives(annotations, full_predictions, assessment_images):
    tot_true_positives = 0
    tot_false_positives = 0
    tot_false_negatives = 0
    for image_name in assessment_images:
        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(full_predictions[image_name]["boxes"])
        pred_scores = np.array(full_predictions[image_name]["scores"])

        sel_pred_boxes = pred_boxes[pred_scores > 0.50]

        num_predicted = sel_pred_boxes.shape[0]
        num_annotated = annotated_boxes.shape[0]

        if num_predicted > 0:
            if num_annotated > 0:
                true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(annotated_boxes, sel_pred_boxes, 0.50)
            else:
                true_positive = 0
                false_positive = num_predicted
                false_negative = 0
        else:
            if num_annotated > 0:
                true_positive = 0
                false_positive = 0
                false_negative = num_annotated
            else:
                true_positive = 0
                false_positive = 0
                false_negative = 0

        tot_true_positives += true_positive
        tot_false_positives += false_positive
        tot_false_negatives += false_negative
    return tot_true_positives, tot_false_positives, tot_false_negatives



def get_accuracies(annotations, full_predictions, assessment_images):
    accuracies = []
    for image_name in assessment_images:
        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(full_predictions[image_name]["boxes"])
        pred_scores = np.array(full_predictions[image_name]["scores"])

        sel_pred_boxes = pred_boxes[pred_scores > 0.50]

        num_predicted = sel_pred_boxes.shape[0]
        num_annotated = annotated_boxes.shape[0]

        if num_predicted > 0:
            if num_annotated > 0:
                true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(annotated_boxes, sel_pred_boxes, 0.50)
                # print(true_positive, false_positive, false_negative)
                precision_050 = true_positive / (true_positive + false_positive)
                recall_050 = true_positive / (true_positive + false_negative)
                if precision_050 == 0 and recall_050 == 0:
                    f1_iou_050 = 0
                else:
                    f1_iou_050 = (2 * precision_050 * recall_050) / (precision_050 + recall_050)
                acc_050 = true_positive / (true_positive + false_positive + false_negative)
                # true_positive, false_positive, false_negative = get_positives_and_negatives(annotated_boxes, sel_region_pred_boxes, 0.75)
                # precision = true_positive / (true_positive + false_positive)
                # recall = true_positive / (true_positive + false_negative)
                # f1_iou_075 = (2 * precision * recall) / (precision + recall)                        

                
            
            else:
                true_positive = 0
                false_positive = num_predicted
                false_negative = 0

                precision_050 = 0.0
                recall_050 = 0.0
                f1_iou_050 = 0.0
                acc_050 = 0.0
        else:
            if num_annotated > 0:
                true_positive = 0
                false_positive = 0
                false_negative = num_annotated

                precision_050 = 0.0
                recall_050 = 0.0
                f1_iou_050 = 0.0
                acc_050 = 0.0
            else:
                true_positive = 0
                false_positive = 0
                false_negative = 0

                precision_050 = 1.0
                recall_050 = 1.0
                f1_iou_050 = 1.0
                acc_050 = 1.0


        accuracies.append(acc_050)
    return accuracies

# (image_set, methods, metric, num_replications, out_dir, xpositions="num_annotations")
# def global_plot(image_set, methods, metric, num_replications, out_dir, xpositions="num_annotations"):

#     chart_data = []
#     for rep_num in range(num_replications):
#         all_training_regions = {}
#         for method in methods:


#             method_label = method["method_label"]
#             image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
#                                         image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
#                                         image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
#                                         image_set["mission_date"])

#             method_image_set = {
#                 "username": image_set["username"],
#                 "farm_name": image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
#                 "field_name": image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
#                 "mission_date": image_set["mission_date"]
#             }


#             recent_result_dir = get_most_recent_result_dir(method_image_set)

#             annotations_path = os.path.join(recent_result_dir, "annotations.json")
#             annotations = annotation_utils.load_annotations(annotations_path)
#             for image_name in annotations.keys():
#                 if image_name not in all_training_regions:
#                     all_training_regions[image_name] = []
#                 all_training_regions[image_name].extend(annotations[image_name]["training_regions"])

#         print("all_training_regions", all_training_regions)
#         for method in methods:

#             chart_entry = {}
#             chart_entry["method_label"] = method["method_label"]
#             chart_entry["vals"] = []
#             image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
#                             image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
#                             image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
#                             image_set["mission_date"])

#             annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
#             annotations = annotation_utils.load_annotations(annotations_path)
#             results_dir = os.path.join(image_set_dir, "model", "results")
#             if os.path.exists(results_dir):
#                 result_dirs = glob.glob(os.path.join(results_dir, "*"))

#                 result_tuples = []
#                 for result_dir in result_dirs:
                    
#                     request_path = os.path.join(result_dir, "request.json")
#                     request = json_io.load_json(request_path)
#                     end_time = request["end_time"]

#                     result_tuples.append((result_dir, end_time))
#                 result_tuples.sort(key=lambda x: x[1])
#                 sorted_result_dirs = [x[0] for x in result_tuples]
#                 for result_dir in sorted_result_dirs:


#                     total_true_positives = 0
#                     total_false_positives = 0
#                     total_false_negatives = 0



#                     predictions_path = os.path.join(result_dir, "predictions.json")
#                     predictions = annotation_utils.load_predictions(predictions_path)

#                     for image_name in annotations.keys():
#                         pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]

#                         sel_pred_boxes = pred_boxes
#                         annotated_boxes = annotations[image_name]["boxes"]

#                         # annotation_inds = box_utils.get_fully_contained_inds(annotations[image_name]["boxes"], all_training_regions[image_name])
#                         # prediction_inds = box_utils.get_fully_contained_inds(pred_boxes, all_training_regions[image_name])
                        
#                         # annotation_mask = np.full(annotations[image_name]["boxes"].shape[0], True)
#                         # annotation_mask[annotation_inds] = False
                        
#                         # pred_mask = np.full(pred_boxes.shape[0], True)
#                         # pred_mask[prediction_inds] = False


#                         # annotated_boxes = annotations[image_name]["boxes"][annotation_mask]
#                         # sel_pred_boxes = pred_boxes[pred_mask]
#                         # acc = get_accuracy(annotated_boxes, sel_pred_boxes)
#                         num_predicted = sel_pred_boxes.shape[0]
#                         num_annotated = annotated_boxes.shape[0]

#                         if num_predicted > 0:
#                             if num_annotated > 0:
#                                 true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(annotated_boxes, sel_pred_boxes, 0.50)
#                             else:
#                                 true_positive = 0
#                                 false_positive = num_predicted
#                                 false_negative = 0
#                         else:
#                             if num_annotated > 0:
#                                 true_positive = 0
#                                 false_positive = 0
#                                 false_negative = num_annotated
#                             else:
#                                 true_positive = 0
#                                 false_positive = 0
#                                 false_negative = 0
#                         total_true_positives += true_positive
#                         total_false_positives += false_positive
#                         total_false_negatives += false_negative

#                     if metric == "accuracy":
#                         global_accuracy = total_true_positives / (total_true_positives + total_false_positives + total_false_negatives)
#                         chart_entry["vals"].append(global_accuracy)
#                     elif metric == "true_positives":
#                         chart_entry["vals"].append(total_true_positives)

#             chart_data.append(chart_entry)


#         fig = plt.figure(figsize=(16, 8))
#         ax = fig.add_subplot(111)
#         colors = ["red", "blue", "green", "purple", "grey", "pink", "yellow"]
#         for i, chart_entry in enumerate(chart_data):
#             method_label = chart_entry["method_label"]
#             color = colors[i]
#             ax.plot([x for x in range(len(chart_entry["vals"]))], chart_entry["vals"], color=color, marker="x", label=method_label)


#         ax.set_ylabel(metric) #"Accuracy") # Accuracy")
#         # if metric == "accuracy": # or metric == "global_accuracy":
#         ax.set_ylim((0, 1))


#         ax.set_xlabel("Iterations") #Number of Annotations Used For Fine-Tuning")

#         ax.legend()
#         plt.tight_layout()
#         if not os.path.exists(os.path.dirname(out_path)):
#             os.makedirs(os.path.dirname(out_path))
#         fig.savefig(out_path) #"artificial_ft_1_fine_tuning_method_comparison.svg")





def create_eval_chart_annotations(image_set, methods, metric, num_replications, out_path):

    chart_data = []
    for method in methods:
        for rep_num in range(num_replications):
            chart_entry = {}
            print("processing", method["method_name"])
            # image_set = method["image_set"]
            method_label = method["method_label"]
            image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                        image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
                                        image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
                                        image_set["mission_date"])

            chart_entry["method_label"] = method_label
            chart_entry["rep_num"] = rep_num


            # image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
            # image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
            # annotations = json_io.load_json(
            #                     os.path.join(image_set_dir, "annotations", "annotations.json")
            # )
            # metadata = json_io.load_json(
            #                 os.path.join(image_set_dir, "metadata", "metadata.json")
            # )

            results_dir = os.path.join(image_set_dir, "model", "results")
            if os.path.exists(results_dir):
                result_dirs = glob.glob(os.path.join(results_dir, "*"))

                # sorted_result_dirs = natsorted(result_dirs, key=lambda y: y.lower())


                # result_tuples = []
                # for result_dir in result_dirs:
                #     annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                #     test_images = []
                #     for image_name in annotations.keys():
                #         if annotation_utils.is_fully_annotated_for_testing(annotations, image_name, 
                #                                     metadata["images"][image_name]["width_px"], metadata["images"][image_name]["height_px"]):
                #             test_images.append(image_name)
                #     result_tuples.append((test_images, result_dir))
                # result_tuples.sort(key=lambda x: len(x[0]))

                # assessment_images = result_tuples[-1][0]
                # result_tuples = []

                chart_entry["mean_vals"] = []
                chart_entry["min_vals"] = []
                chart_entry["max_vals"] = []
                chart_entry["num_annotations"] = []
                for result_dir in result_dirs:
                    print("\tprocessing", os.path.basename(result_dir))

                    annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                    # num_training_images = 0
                    num_annotations = 0
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["training_regions"]) > 0:
                            # num_training_images += 1
                            # for region in annotations[image_name]["training_region"]:
                            num_annotations += (box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"])).size

                    # result_tuples.append((num_annotations, result_dir))
                    # result_tuples.sort(key=lambda x: x[0])

                    # method["mean_accuracies"] = []
                    # for result_tuple in result_tuples:

                    full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                    full_predictions = json_io.load_json(full_predictions_path)
                    if metric == "accuracy":
                        vals = get_accuracies(annotations, full_predictions, list(annotations.keys()))
                    elif metric == "percent_count_error":
                        vals = get_percent_count_errors(annotations, full_predictions, list(annotations.keys()))
                    elif metric == "global_accuracy":
                        global_accuracy = get_global_accuracy(annotations, full_predictions, list(annotations.keys()))
                    else:
                        vals = get_dics(annotations, full_predictions, list(annotations.keys()))
                    # df = pd.read_excel(os.path.join(result_dir, "metrics.xlsx"))

                    # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
                    # subset_df = df.iloc[included_rows]

                    # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])
                    # mean_accuracy = np.mean(df["Accuracy (IoU=.50, conf>.50)"])
                    if metric != "global_accuracy":
                        chart_entry["mean_vals"].append(np.mean(vals))
                        chart_entry["min_vals"].append(np.min(vals))
                        chart_entry["max_vals"].append(np.max(vals))
                    else:
                        chart_entry["mean_vals"].append(global_accuracy)
                    chart_entry["num_annotations"].append(num_annotations)

                order = np.argsort(chart_entry["num_annotations"])
                chart_entry["mean_vals"] = np.array(chart_entry["mean_vals"])[order]
                if metric != "global_accuracy":
                    chart_entry["min_vals"] = np.array(chart_entry["min_vals"])[order]
                    chart_entry["max_vals"] = np.array(chart_entry["max_vals"])[order]
                chart_entry["num_annotations"] = np.array(chart_entry["num_annotations"])[order]

                chart_data.append(chart_entry)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    colors = ["red", "blue", "green", "purple", "grey", "pink", "yellow"]
    color_index = 0
    method_label_to_color = {}
    for chart_entry in chart_data:
        # print(method)
        method_label = chart_entry["method_label"] #+ "-" + chart_entry["rep_num"]
        if method_label not in method_label_to_color:
            method_label_to_color[method_label] = colors[color_index]
            color_index += 1
        color = method_label_to_color[method_label]

        if chart_entry["rep_num"] == 0:
            ax.plot(chart_entry["num_annotations"], chart_entry["mean_vals"], color=color, marker="x", label=method_label, alpha=0.9) #, alpha=0.75, label=method_name)
        else:
            ax.plot(chart_entry["num_annotations"], chart_entry["mean_vals"], color=color, marker="x", alpha=0.9)
        
        if metric != "global_accuracy":
            ax.plot(chart_entry["num_annotations"], chart_entry["min_vals"], color=color, marker="x", linestyle="dashed", alpha=0.9) #, alpha=0.75, linestyle='dashed')
            ax.plot(chart_entry["num_annotations"], chart_entry["max_vals"], color=color, marker="x", linestyle="dashed", alpha=0.9) #, linestyle='dashed')
        
    ax.set_ylabel(metric.capitalize()) # Accuracy")
    if metric == "accuracy": # or metric == "global_accuracy":
        ax.set_ylim((0, 1))


    ax.set_xlabel("Number of Annotations Used For Fine-Tuning")

    ax.legend()
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    fig.savefig(out_path) #"artificial_ft_1_fine_tuning_method_comparison.svg")


def create_eval_chart_annotations_2(image_set, methods, metric, num_replications, out_path):

    chart_data = []
    for method in methods:
        for rep_num in range(num_replications):
            chart_entry = {}
            print("processing", method["method_name"])
            # image_set = method["image_set"]
            method_label = method["method_label"]
            image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                        image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
                                        image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
                                        image_set["mission_date"])

            chart_entry["method_label"] = method_label
            chart_entry["rep_num"] = rep_num


            # image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
            # image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
            # annotations = json_io.load_json(
            #                     os.path.join(image_set_dir, "annotations", "annotations.json")
            # )
            # metadata = json_io.load_json(
            #                 os.path.join(image_set_dir, "metadata", "metadata.json")
            # )

            results_dir = os.path.join(image_set_dir, "model", "results")
            if os.path.exists(results_dir):
                result_dirs = glob.glob(os.path.join(results_dir, "*"))

                # sorted_result_dirs = natsorted(result_dirs, key=lambda y: y.lower())


                # result_tuples = []
                # for result_dir in result_dirs:
                #     annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                #     test_images = []
                #     for image_name in annotations.keys():
                #         if annotation_utils.is_fully_annotated_for_testing(annotations, image_name, 
                #                                     metadata["images"][image_name]["width_px"], metadata["images"][image_name]["height_px"]):
                #             test_images.append(image_name)
                #     result_tuples.append((test_images, result_dir))
                # result_tuples.sort(key=lambda x: len(x[0]))

                # assessment_images = result_tuples[-1][0]
                # result_tuples = []

                chart_entry["mean_vals"] = []
                chart_entry["min_vals"] = []
                chart_entry["max_vals"] = []
                chart_entry["num_annotations"] = []
                for result_dir in result_dirs:
                    print("\tprocessing", os.path.basename(result_dir))

                    annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                    # num_training_images = 0
                    num_annotations = 0
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["training_regions"]) > 0:
                            # num_training_images += 1
                            # for region in annotations[image_name]["training_region"]:
                            num_annotations += (box_utils.get_fully_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"])).size

                    # result_tuples.append((num_annotations, result_dir))
                    # result_tuples.sort(key=lambda x: x[0])

                    # method["mean_accuracies"] = []
                    # for result_tuple in result_tuples:

                    full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                    full_predictions = json_io.load_json(full_predictions_path)
                    if metric == "accuracy":
                        vals = get_accuracies(annotations, full_predictions, list(annotations.keys()))
                    elif metric == "percent_count_error":
                        vals = get_percent_count_errors(annotations, full_predictions, list(annotations.keys()))
                    else:
                        vals = get_dics(annotations, full_predictions, list(annotations.keys()))
                    # df = pd.read_excel(os.path.join(result_dir, "metrics.xlsx"))

                    # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
                    # subset_df = df.iloc[included_rows]

                    # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])
                    # mean_accuracy = np.mean(df["Accuracy (IoU=.50, conf>.50)"])

                    chart_entry["mean_vals"].append(np.mean(vals))
                    chart_entry["min_vals"].append(np.min(vals))
                    chart_entry["max_vals"].append(np.max(vals))
                    chart_entry["num_annotations"].append(num_annotations)

                order = np.argsort(chart_entry["num_annotations"])
                chart_entry["mean_vals"] = np.array(chart_entry["mean_vals"])[order]
                chart_entry["min_vals"] = np.array(chart_entry["min_vals"])[order]
                chart_entry["max_vals"] = np.array(chart_entry["max_vals"])[order]
                chart_entry["num_annotations"] = np.array(chart_entry["num_annotations"])[order]

                chart_data.append(chart_entry)

    accuracies = np.arange(0, 1, 0.05) #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #np.arange(0, 100, 10)
    final_chart_data = {}
    for chart_entry in chart_data:
        method_label = chart_entry["method_label"]
        if method_label not in final_chart_data:
            final_chart_data[method_label] = {}
        for accuracy in accuracies:
            valid_entries = chart_entry["min_vals"] >= accuracy
            if np.all(valid_entries == False):
                continue
            min_needed_to_achieve_accuracy = np.min(chart_entry["num_annotations"][valid_entries])
            if accuracy not in final_chart_data[method_label]:
                final_chart_data[method_label][accuracy] = []
            final_chart_data[method_label][accuracy].append(min_needed_to_achieve_accuracy)
    plotted_chart_data = {}
    for method_label in final_chart_data.keys():
        plotted_chart_data[method_label] = {}
        plotted_chart_data[method_label]["accuracies"] = sorted(list(final_chart_data[method_label].keys()))
        kept_accuracies = []
        for accuracy in plotted_chart_data[method_label]["accuracies"]:
            if len(final_chart_data[method_label][accuracy]) == num_replications:
                kept_accuracies.append(accuracy)
        plotted_chart_data[method_label]["accuracies"] = kept_accuracies
        plotted_chart_data[method_label]["num_needed"] = []
        for accuracy in plotted_chart_data[method_label]["accuracies"]:
            plotted_chart_data[method_label]["num_needed"].append(np.mean(final_chart_data[method_label][accuracy]))


    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    color_index = 0
    method_label_to_color = {}
    for i, method_label in enumerate(plotted_chart_data.keys()):
        color = colors[i]
        ax.plot(plotted_chart_data[method_label]["num_needed"], plotted_chart_data[method_label]["accuracies"], color=color, marker="o", label=method_label, linestyle="dashed") #, alpha=0.75, label=method_name)

    # for chart_entry in chart_data:
    #     # print(method)
    #     method_label = chart_entry["method_label"] #+ "-" + chart_entry["rep_num"]
    #     if method_label not in method_label_to_color:
    #         method_label_to_color[method_label] = colors[color_index]
    #         color_index += 1
    #     color = method_label_to_color[method_label]

    #     # if chart_entry["rep_num"] == 0:
    #     #     ax.plot(chart_entry["num_annotations"], chart_entry["mean_vals"], color=color, marker="x", label=method_label, alpha=0.9) #, alpha=0.75, label=method_name)
    #     # else:
    #     #     ax.plot(chart_entry["num_annotations"], chart_entry["mean_vals"], color=color, marker="x", alpha=0.9)
        
    #     # ax.plot(chart_entry["num_annotations"], chart_entry["min_vals"], color=color, marker="x", linestyle="dashed", alpha=0.9) #, alpha=0.75, linestyle='dashed')
    #     # ax.plot(chart_entry["num_annotations"], chart_entry["max_vals"], color=color, marker="x", linestyle="dashed", alpha=0.9) #, linestyle='dashed')
    
    # ax.set_ylabel("Mean " + metric.capitalize()) # Accuracy")
    ax.set_ylabel("Worst-Case " + metric.capitalize())
    if metric == "accuracy":
        ax.set_ylim((0, 1))

    ax.set_xlabel("Average Number of Annotations Needed") #Number of Annotations Used For Fine-Tuning")
    # for accuracy in accuracies:
    #     ax.axhline(accuracy, linestyle="dashed", color="grey")

    ax.legend()
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    fig.savefig(out_path) #"artificial_ft_1_fine_tuning_method_comparison.svg")


# def create_eval_chart(methods, out_path):
#     # dfs = {}
#     excluded = []

#     # for method in methods:
#     #     image_set = method["image_set"]
#     #     training_image_selection_method = method["training_image_selection_method"]
#     #     image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
#     #                                 image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

#     #     results_dir = os.path.join(image_set_dir, "model", "results")
#     #     result_dirs = glob.glob(os.path.join(results_dir, "*"))
#     #     result_tuples = []
#     #     for result_dir in result_dirs:
#     #         annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
#     #         num_training_images = 0
#     #         for image_name in annotations.keys():
#     #             if len(annotations[image_name]["training_regions"]) > 0:
#     #                 num_training_images += 1
#     #         result_tuples.append((num_training_images, result_dir))
#     #     result_tuples.sort(key=lambda x: x[0])
#     #     last_result_dir = result_tuples[-1][1]

#     #     df = pd.read_excel(os.path.join(last_result_dir, "metrics.xlsx"))

#     #     excluded.extend(df.index[df["Image Is Fully Annotated"] == "yes: for fine-tuning"].tolist())
#     #     # dfs[training_image_selection_method] = df
#     # print("{} images are excluded because they were used for testing.".format((np.unique(excluded)).size))
    


#     for method in methods:
#         image_set = method["image_set"]
#         training_image_selection_method = method["training_image_selection_method"]
#         image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
#                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

#         results_dir = os.path.join(image_set_dir, "model", "results")
#         result_dirs = glob.glob(os.path.join(results_dir, "*"))
#         result_tuples = []
#         for result_dir in result_dirs:
#             annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
#             num_training_images = 0
#             for image_name in annotations.keys():
#                 if len(annotations[image_name]["training_regions"]) > 0:
#                     num_training_images += 1
#             result_tuples.append((num_training_images, result_dir))
#         result_tuples.sort(key=lambda x: x[0])

#         method["mean_accuracies"] = []
#         for result_tuple in result_tuples:
#             df = pd.read_excel(os.path.join(result_tuple[1], "metrics.xlsx"))

#             # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
#             # subset_df = df.iloc[included_rows]

#             # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])
#             mean_accuracy = np.mean(df["Accuracy (IoU=.50, conf>.50)"])

#             method["mean_accuracies"].append(mean_accuracy)

#     fig = plt.figure(figsize=(16, 8))
#     ax = fig.add_subplot(111)
#     for method in methods:
#         print(method)
#         training_image_selection_method = method["training_image_selection_method"]
#         ax.plot(np.arange(0, len(method["mean_accuracies"])), method["mean_accuracies"], label=training_image_selection_method)
    
#     ax.set_ylabel("Mean Accuracy")
#     ax.set_ylim((0, 1))

#     ax.set_xlabel("Number of Images Used For Fine-Tuning")

#     ax.legend()
#     plt.tight_layout()
#     fig.savefig(out_path) #"artificial_ft_1_fine_tuning_method_comparison.svg")



def create_boxplot_comparison(image_set, methods, metric, num_replications, out_dir, xpositions="num_annotations"):
    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
            
        # use plot function to draw a small line to name the legend.
        plt.plot([], c=color_code, label=label)
        plt.legend()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for rep_num in range(num_replications):
        chart_data = []
        for method in methods:
        
            chart_entry = {}
            print("processing", method["method_name"])
            # image_set = method["image_set"]
            method_label = method["method_label"]
            image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                        image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
                                        image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
                                        image_set["mission_date"])

            chart_entry["method_label"] = method_label
            chart_entry["rep_num"] = rep_num


            # image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
            # image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
            # annotations = json_io.load_json(
            #                     os.path.join(image_set_dir, "annotations", "annotations.json")
            # )
            # metadata = json_io.load_json(
            #                 os.path.join(image_set_dir, "metadata", "metadata.json")
            # )

            results_dir = os.path.join(image_set_dir, "model", "results")
            if os.path.exists(results_dir):
                result_dirs = glob.glob(os.path.join(results_dir, "*"))

                # sorted_result_dirs = natsorted(result_dirs, key=lambda y: y.lower())


                # result_tuples = []
                # for result_dir in result_dirs:
                #     annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                #     test_images = []
                #     for image_name in annotations.keys():
                #         if annotation_utils.is_fully_annotated_for_testing(annotations, image_name, 
                #                                     metadata["images"][image_name]["width_px"], metadata["images"][image_name]["height_px"]):
                #             test_images.append(image_name)
                #     result_tuples.append((test_images, result_dir))
                # result_tuples.sort(key=lambda x: len(x[0]))

                # assessment_images = result_tuples[-1][0]
                # result_tuples = []


                chart_entry["vals"] = []
                chart_entry["num_annotations"] = []
                for result_dir in result_dirs:
                    print("\tprocessing", os.path.basename(result_dir))

                    annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                    # num_training_images = 0
                    num_annotations = 0
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["training_regions"]) > 0:
                            # num_training_images += 1
                            # for region in annotations[image_name]["training_region"]:
                            num_annotations += (box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"])).size

                    # result_tuples.append((num_annotations, result_dir))
                    # result_tuples.sort(key=lambda x: x[0])

                    # method["mean_accuracies"] = []
                    # for result_tuple in result_tuples:

                    full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                    full_predictions = json_io.load_json(full_predictions_path)
                    if metric == "accuracy":
                        vals = get_accuracies(annotations, full_predictions, list(annotations.keys()))
                    elif metric == "percent_count_error":
                        vals = get_percent_count_errors(annotations, full_predictions, list(annotations.keys()))
                    else:
                        vals = get_dics(annotations, full_predictions, list(annotations.keys()))
                    # df = pd.read_excel(os.path.join(result_dir, "metrics.xlsx"))

                    # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
                    # subset_df = df.iloc[included_rows]

                    # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])
                    # mean_accuracy = np.mean(df["Accuracy (IoU=.50, conf>.50)"])

                    # chart_entry["mean_vals"].append(np.mean(vals))
                    # chart_entry["min_vals"].append(np.min(vals))
                    # chart_entry["max_vals"].append(np.max(vals))
                    chart_entry["vals"].append(vals)
                    chart_entry["num_annotations"].append(num_annotations)

                order = np.argsort(chart_entry["num_annotations"])
                chart_entry["vals"] = (np.array(chart_entry["vals"])[order]).tolist()
                chart_entry["num_annotations"] = (np.array(chart_entry["num_annotations"])[order]).tolist()

                chart_data.append(chart_entry)

        fig = plt.figure(figsize=(16, 8))
        # ax = fig.add_subplot(111)
        min_val = np.inf
        max_val = (-1) * np.inf
        print("len chart_data", len(chart_data))
        for i, chart_entry in enumerate(chart_data):
            print("len num_annotations", len(chart_entry["num_annotations"]))
            print("len chart_entry vals", len(chart_entry["vals"]))
            # if i == 0:
            #     widths = 0.8
            # else:
            
            if xpositions == "num_annotations":
                positions = chart_entry["num_annotations"]
                widths = 1.5
            else:
                positions = [i for i in range(len(chart_entry["vals"]))]
                widths = 0.1
            chart_entry_plot = plt.boxplot(chart_entry["vals"], positions=positions, notch=True, widths=widths, whis=(0, 100))
            define_box_properties(chart_entry_plot, colors[i], chart_entry["method_label"])

            cur_min = np.min(chart_entry["vals"])
            if cur_min < min_val:
                min_val = cur_min

            cur_max = np.max(chart_entry["vals"])
            if cur_max > max_val:
                max_val = cur_max



            # ticks = np.arange(0, len(methods[0]["accuracies"]))
            # set the x label values
            # plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
            
            # set the limit for x axis
            # plt.xlim(-2, len(ticks)*2)
            
            # set the limit for y axis
        if metric == "accuracy":
            plt.ylim(0, 1.0)
        if metric == "dic":
            plt.axhline(0, color="black", linestyle="dashed", linewidth=1)
            ext_val = max(abs(max_val), abs(min_val))
            plt.ylim(-ext_val, ext_val)
        
        plt.xlabel(xpositions)
        plt.ylabel(metric)
        # set the title
        #plt.title('Grouped boxplot using matplotlib')
        chart_out_dir = os.path.join(out_dir, metric, xpositions)
        os.makedirs(chart_out_dir, exist_ok=True)
        plt.savefig(os.path.join(chart_out_dir, "rep_" + str(rep_num) + ".svg"))



def create_global_comparison(image_set, methods, metric, num_replications, out_dir, xpositions="num_annotations"):

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    chart_data = {}
    for rep_num in range(num_replications):
        chart_data[rep_num] = []
        for method in methods:
        
            chart_entry = {}
            print("processing", method["method_name"])
            method_label = method["method_label"]
            image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                        image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
                                        image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
                                        image_set["mission_date"])

            chart_entry["method_label"] = method_label
            chart_entry["rep_num"] = rep_num

            results_dir = os.path.join(image_set_dir, "model", "results")
            if os.path.exists(results_dir):
                result_dirs = glob.glob(os.path.join(results_dir, "*"))

                chart_entry["vals"] = []
                chart_entry["num_annotations"] = []
                for result_dir in result_dirs:
                    print("\tprocessing", os.path.basename(result_dir))

                    annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                    # num_training_images = 0
                    num_annotations = 0
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["training_regions"]) > 0:
                            num_annotations += (box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"])).size


                    full_predictions_path = os.path.join(result_dir, "predictions.json")
                    full_predictions = annotation_utils.load_predictions(full_predictions_path) #json_io.load_json(full_predictions_path)

                    



                    if metric == "accuracy":
                        num_true_positives, num_false_positives, num_false_negatives = get_positives_and_negatives(annotations, full_predictions, list(annotations.keys()))
                        val = num_true_positives / (num_true_positives + num_false_positives + num_false_negatives)
                        # vals = get_accuracies(annotations, full_predictions, list(annotations.keys()))
                    elif metric == "AP (IoU=.50:.05:.95)":
                        val = get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                    elif metric == "AP (IoU=.50)":
                        val = get_AP(annotations, full_predictions, iou_thresh=".50")

                    elif metric == "AP (IoU=.75)":
                        val = get_AP(annotations, full_predictions, iou_thresh=".75")
                    # elif metric == "true_positives":
                    #     val = num_true_positives
                    #     vals = get_percent_count_errors(annotations, full_predictions, list(annotations.keys()))

                    # else:
                    #     vals = get_dics(annotations, full_predictions, list(annotations.keys()))
                    # df = pd.read_excel(os.path.join(result_dir, "metrics.xlsx"))

                    # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
                    # subset_df = df.iloc[included_rows]

                    # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])
                    # mean_accuracy = np.mean(df["Accuracy (IoU=.50, conf>.50)"])

                    # chart_entry["mean_vals"].append(np.mean(vals))
                    # chart_entry["min_vals"].append(np.min(vals))
                    # chart_entry["max_vals"].append(np.max(vals))
                    chart_entry["vals"].append(val)
                    chart_entry["num_annotations"].append(num_annotations)

                order = np.argsort(chart_entry["num_annotations"])
                chart_entry["vals"] = (np.array(chart_entry["vals"])[order]).tolist()
                chart_entry["num_annotations"] = (np.array(chart_entry["num_annotations"])[order]).tolist()

                chart_data[rep_num].append(chart_entry)

    fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(111)
    min_val =  10000000 #np.inf
    max_val = -10000000 #(-1) * np.inf
    # print("len chart_data", len(chart_data))
    for rep_num in chart_data.keys():
        for i, chart_entry in enumerate(chart_data[rep_num]):
            # print("len num_annotations", len(chart_entry["num_annotations"]))
            # print("len chart_entry vals", len(chart_entry["vals"]))
            # if i == 0:
            #     widths = 0.8
            # else:
            
            if xpositions == "num_annotations":
                positions = chart_entry["num_annotations"]
            else:
                positions = [i for i in range(len(chart_entry["vals"]))]
            # for col in range(len(chart_entry["vals"][0])):
            #     # if col == 0:
                    
            #     # else:
            #     #     label = None
            #     plt.plot(positions, np.array(chart_entry["vals"])[:, col], color=colors[i], linewidth=1, alpha=0.3) #, label=label)
            # label = chart_entry["method_label"]
            if rep_num == 0:
                label = chart_entry["method_label"]
            else:
                label = None
            plt.plot(positions, chart_entry["vals"], color=colors[i], linewidth=1, label=label, marker="o")
            # chart_entry_plot = plt.boxplot(chart_entry["vals"], positions=positions, notch=True, widths=widths, whis=(0, 100))
            # define_box_properties(chart_entry_plot, colors[i], chart_entry["method_label"])

            # cur_min = np.min(chart_entry["vals"])
            # if cur_min < min_val:
            #     min_val = cur_min

            # cur_max = np.max(chart_entry["vals"])
            # if cur_max > max_val:
            #     max_val = cur_max



            # ticks = np.arange(0, len(methods[0]["accuracies"]))
            # set the x label values
            # plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
            
            # set the limit for x axis
            # plt.xlim(-2, len(ticks)*2)
            
            # set the limit for y axis
    # if metric == "accuracy":
    #     plt.ylim(0, 1.0)
    # if metric == "dic":
    #     plt.axhline(0, color="black", linestyle="dashed", linewidth=1)
    #     ext_val = max(abs(max_val), abs(min_val))
    #     plt.ylim(-ext_val, ext_val)
    
    plt.legend()
    plt.xlabel(xpositions)
    plt.ylabel(metric)
    # set the title
    #plt.title('Grouped boxplot using matplotlib')
    chart_out_dir = os.path.join(out_dir, metric, xpositions)
    os.makedirs(chart_out_dir, exist_ok=True)
    plt.savefig(os.path.join(chart_out_dir, "all_replications.svg"))







def create_thinline_comparison(image_set, methods, metric, num_replications, out_dir, xpositions="num_annotations", include_mean_line=True):

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for rep_num in range(num_replications):
        chart_data = []
        for method in methods:
        
            chart_entry = {}
            print("processing", method["method_name"])
            # image_set = method["image_set"]
            method_label = method["method_label"]
            image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                        image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num),
                                        image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(rep_num), 
                                        image_set["mission_date"])

            chart_entry["method_label"] = method_label
            chart_entry["rep_num"] = rep_num


            # image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
            # image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
            # annotations = json_io.load_json(
            #                     os.path.join(image_set_dir, "annotations", "annotations.json")
            # )
            # metadata = json_io.load_json(
            #                 os.path.join(image_set_dir, "metadata", "metadata.json")
            # )

            results_dir = os.path.join(image_set_dir, "model", "results")
            if os.path.exists(results_dir):
                result_dirs = glob.glob(os.path.join(results_dir, "*"))

                # sorted_result_dirs = natsorted(result_dirs, key=lambda y: y.lower())


                # result_tuples = []
                # for result_dir in result_dirs:
                #     annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                #     test_images = []
                #     for image_name in annotations.keys():
                #         if annotation_utils.is_fully_annotated_for_testing(annotations, image_name, 
                #                                     metadata["images"][image_name]["width_px"], metadata["images"][image_name]["height_px"]):
                #             test_images.append(image_name)
                #     result_tuples.append((test_images, result_dir))
                # result_tuples.sort(key=lambda x: len(x[0]))

                # assessment_images = result_tuples[-1][0]
                # result_tuples = []


                chart_entry["vals"] = []
                chart_entry["num_annotations"] = []
                for result_dir in result_dirs:
                    print("\tprocessing", os.path.basename(result_dir))

                    annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
                    # num_training_images = 0
                    num_annotations = 0
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["training_regions"]) > 0:
                            # num_training_images += 1
                            # for region in annotations[image_name]["training_region"]:
                            num_annotations += (box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"])).size

                    # result_tuples.append((num_annotations, result_dir))
                    # result_tuples.sort(key=lambda x: x[0])

                    # method["mean_accuracies"] = []
                    # for result_tuple in result_tuples:

                    full_predictions_path = os.path.join(result_dir, "predictions.json")
                    full_predictions = json_io.load_json(full_predictions_path)
                    if metric == "accuracy":
                        vals = get_accuracies(annotations, full_predictions, list(annotations.keys()))
                    elif metric == "percent_count_error":
                        vals = get_percent_count_errors(annotations, full_predictions, list(annotations.keys()))
                    elif metric == "abs_dic":
                        vals = np.abs(get_dics(annotations, full_predictions, list(annotations.keys())))
                    elif metric == "mse":
                        vals = np.array(get_dics(annotations, full_predictions, list(annotations.keys()))) ** 2
                    else:
                        vals = get_dics(annotations, full_predictions, list(annotations.keys()))

                    # df = pd.read_excel(os.path.join(result_dir, "metrics.xlsx"))

                    # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
                    # subset_df = df.iloc[included_rows]

                    # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])
                    # mean_accuracy = np.mean(df["Accuracy (IoU=.50, conf>.50)"])

                    # chart_entry["mean_vals"].append(np.mean(vals))
                    # chart_entry["min_vals"].append(np.min(vals))
                    # chart_entry["max_vals"].append(np.max(vals))
                    chart_entry["vals"].append(vals)
                    chart_entry["num_annotations"].append(num_annotations)

                order = np.argsort(chart_entry["num_annotations"])
                chart_entry["vals"] = (np.array(chart_entry["vals"])[order]).tolist()
                chart_entry["num_annotations"] = (np.array(chart_entry["num_annotations"])[order]).tolist()

                chart_data.append(chart_entry)

        fig = plt.figure(figsize=(16, 8))
        # ax = fig.add_subplot(111)
        min_val =  10000000 #np.inf
        max_val = -10000000 #(-1) * np.inf
        print("len chart_data", len(chart_data))
        for i, chart_entry in enumerate(chart_data):
            print("len num_annotations", len(chart_entry["num_annotations"]))
            print("len chart_entry vals", len(chart_entry["vals"]))
            # if i == 0:
            #     widths = 0.8
            # else:
            
            if xpositions == "num_annotations":
                positions = chart_entry["num_annotations"]
                widths = 1.5
            else:
                positions = [i for i in range(len(chart_entry["vals"]))]
                widths = 0.1
            for col in range(len(chart_entry["vals"][0])):
                if col == 0:
                    label = chart_entry["method_label"]
                else:
                    label = None
                plt.plot(positions, np.array(chart_entry["vals"])[:, col], color=colors[i], linewidth=1, alpha=0.3, label=label)
            # label = chart_entry["method_label"]
            if include_mean_line:
                plt.plot(positions, np.mean(np.array(chart_entry["vals"]), axis=1), color=colors[i], linewidth=2)
            # chart_entry_plot = plt.boxplot(chart_entry["vals"], positions=positions, notch=True, widths=widths, whis=(0, 100))
            # define_box_properties(chart_entry_plot, colors[i], chart_entry["method_label"])

            cur_min = np.min(chart_entry["vals"])
            if cur_min < min_val:
                min_val = cur_min

            cur_max = np.max(chart_entry["vals"])
            if cur_max > max_val:
                max_val = cur_max



            # ticks = np.arange(0, len(methods[0]["accuracies"]))
            # set the x label values
            # plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
            
            # set the limit for x axis
            # plt.xlim(-2, len(ticks)*2)
            
            # set the limit for y axis
        if metric == "accuracy":
            plt.ylim(0, 1.0)
        if metric == "dic":
            plt.axhline(0, color="black", linestyle="dashed", linewidth=1)
            ext_val = max(abs(max_val), abs(min_val))
            plt.ylim(-ext_val, ext_val)
        
        plt.legend()
        plt.xlabel(xpositions)
        plt.ylabel(metric)
        # set the title
        #plt.title('Grouped boxplot using matplotlib')
        chart_out_dir = os.path.join(out_dir, metric, xpositions)
        os.makedirs(chart_out_dir, exist_ok=True)
        plt.savefig(os.path.join(chart_out_dir, "rep_" + str(rep_num) + ".svg"))



def get_most_recent_result_dir(image_set):
    image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
        image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

    result_pairs = []
    results_dir = os.path.join(image_set_dir, "model", "results")
    for result_dir in glob.glob(os.path.join(results_dir, "*")):
        request_path = os.path.join(result_dir, "request.json")
        request = json_io.load_json(request_path)
        end_time = request["end_time"]
        result_pairs.append((result_dir, end_time))

    result_pairs.sort(key=lambda x: x[1])
    return result_pairs[-1][0]


def select_training_images(image_set, method, num):
    image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
        image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
    annotations = annotation_utils.load_annotations(
                        os.path.join(image_set_dir, "annotations", "annotations.json")
    )
    
    if method == "random":
        candidates = []
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                candidates.append(image_name)
        chosen_image_names = random.sample(candidates, min(num, len(candidates))) #random.choice(candidates)
        return chosen_image_names
    elif method == "lowest_quality" or method == "highest_quality":
        results_dir = os.path.join(image_set_dir, "model", "results")
        result_dirs = glob.glob(os.path.join(results_dir, "*"))
        result_tuples = []
        for result_dir in result_dirs:
            annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
            num_training_images = 0
            for image_name in annotations.keys():
                if len(annotations[image_name]["training_regions"]) > 0:
                    num_training_images += 1
            result_tuples.append((num_training_images, result_dir))
        result_tuples.sort(key=lambda x: x[0])
        last_result_dir = result_tuples[-1][1]
        predictions = json_io.load_json(os.path.join(last_result_dir, "predictions.json"))

        annotations = annotation_utils.load_annotations(
                            os.path.join(image_set_dir, "annotations", "annotations.json")
        )
        quality_tuples = []
        for image_name in predictions.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                quality = get_confidence_quality(np.array(predictions[image_name]["scores"]))
                quality_tuples.append((quality, image_name))
        quality_tuples.sort(key=lambda x: x[0])
        if method == "lowest_quality":
            # print("choosing {} (quality: {})".format(quality_tuples[0][1], quality_tuples[0][0]))
            chosen_image_names = []
            for tup in quality_tuples[:num]:
                chosen_image_names.append(tup[1])
            return chosen_image_names
            
            # return quality_tuples[0][1]
        elif method == "highest_quality":
            chosen_image_names = []
            for tup in quality_tuples[-1 * num:]:
                chosen_image_names.append(tup[1])
            return chosen_image_names
            # return quality_tuples[-1][1]

def add_training_annotations(image_set, method):

    image_based_methods = ["rand_img", "sel_img", "sel_worst_img"]
    region_based_methods = ["rand_img_rand_reg", "sel_img_rand_reg", "sel_img_sel_reg", "img_split", "quartile", "regions_match_image_anno_count", "low_quality_regions_match_image_anno_count"]

    image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
        image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    last_result_dir = get_most_recent_result_dir(image_set)

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)

    predictions_path = os.path.join(last_result_dir, "predictions.json")
    predictions = json_io.load_json(predictions_path)



    if method["method_name"] in image_based_methods: #== "rand_img" or method["method_name"] == "sel_img":
        num_images = method["num_images"]

        candidates = []
        for image_name in annotations.keys():
            if len(annotations[image_name]["training_regions"]) == 0:
                candidates.append(image_name)

        if method["method_name"] == "rand_img":
            chosen_images = random.sample(candidates, num_images)
        elif method["method_name"] == "sel_img" or method["method_name"] == "sel_worst_img":
            quality_tuples = []
            for image_name in candidates:
                quality = get_confidence_quality(np.array(predictions[image_name]["scores"]))
                quality_tuples.append((quality, image_name))
            quality_tuples.sort(key=lambda x: x[0])
            if method["method_name"] == "sel_img":
                chosen_images = [x[1] for x in quality_tuples[:num_images]]
            else:
                chosen_images = [x[1] for x in quality_tuples[-num_images:]]

        for image_name in chosen_images:
            annotations[image_name]["training_regions"].append([
                0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]
            ])


    elif method["method_name"] in region_based_methods:
    

        
        if method["method_name"] == "rand_img_rand_reg":

            num_regions = method["num_regions"]
            num_annotations_per_region = method["num_annotations_per_region"]
            # for i in range(num_regions):
            image_names = list(annotations.keys())
            random.shuffle(image_names)
            num_added = 0
            index = 0
            while num_added < num_regions:
                if index > len(image_names):
                    raise RuntimeError("Ran out of choices")
                image_name = image_names[index] #random.choice(annotations.keys())
                image_h = metadata["images"][image_name]["height_px"]
                image_w = metadata["images"][image_name]["width_px"]
                new_region = random_region(annotations, num_annotations_per_region, image_name, image_h, image_w)
                
                if new_region is not None:
                    annotations[image_name]["training_regions"].append(new_region)
                    num_added += 1
                index += 1

        elif method["method_name"] == "img_split" or method["method_name"] == "quartile":
            num_regions_per_image = method["num_regions_per_image"]
            # existing_training_regions = {}
            # for image_name in annotations.keys():
            #     existing_training_regions[image_name] = []
            #     for training_region in annotations[image_name]["training_regions"]:
            #         existing_training_regions[image_name].append(training_region)

            # candidate_regions = []
            quality_tuples = []
            if num_regions_per_image == 4:
                for image_name in annotations.keys():
                    image_h = metadata["images"][image_name]["height_px"]
                    image_w = metadata["images"][image_name]["width_px"]
                    reg_0 = [0,0,image_h//2, image_w//2]
                    reg_1 = [0,(image_w//2),image_h//2,image_w]
                    reg_2 = [(image_h//2),0,image_h,image_w//2]
                    reg_3 = [(image_h//2),(image_w//2),image_h,image_w]

                    regs = [reg_0, reg_1, reg_2, reg_3]
                    for reg in regs:
                        if reg not in annotations[image_name]["training_regions"]: #existing_training_regions[image_name]:
                            inds = box_utils.get_fully_contained_inds(np.array(predictions[image_name]["boxes"]), [reg])
                            # if inds.size > 20:
                            quality = get_confidence_quality(np.array(predictions[image_name]["scores"])[inds])
                            quality_tuples.append((quality, image_name, reg, inds.size))

                            # candidate_regions.append(reg)
                            # quality_tuples = []
                            # for image_name in candidates:
                            #     quality = get_confidence_quality(np.array(predictions[image_name]["scores"]))
                            #     quality_tuples.append((quality, image_name))
                            # quality_tuples.sort(key=lambda x: x[0])
                            # chosen_images = [x[1] for x in quality_tuples[:num_images]]
                # candidate_tuples = []
                # for quality_tuple in quality_tuples:
                #     if quality_tuple[3] >= 20:
                #         candidate_tuples.append(quality_tuple)
                # if len(candidate_tuples) > (method["num_low"] + method["num_mid"] + method["num_high"]):
                #     quality_tuples = candidate_tuples
                            

                if method["method_name"] == "quartile":
                    qualities = np.array([x[0] for x in quality_tuples])
                    image_names = np.array([x[1] for x in quality_tuples])
                    regions = np.array([x[2] for x in quality_tuples])
                    region_sizes = np.array([x[3] for x in quality_tuples])
                    q1, q2, q3 = np.percentile(region_sizes, [25, 50, 75], interpolation="midpoint")
                    q0_data_mask = region_sizes < q1
                    q1_data_mask = np.logical_and(region_sizes >= q1, region_sizes < q2)
                    q2_data_mask = np.logical_and(region_sizes >= q2, region_sizes < q3)
                    q3_data_mask = region_sizes >= q3

                    print("q1, q2, q3", q1, q2, q3)

                    data_masks = [q0_data_mask, q1_data_mask, q2_data_mask, q3_data_mask]
                    num_taken = 0
                    # num_needed_this_iteration = 1
                    for i in range(4):
                        print("iteration", i)
                        print("number of items in this quantile", np.sum(data_masks[i]))
                        num_needed_this_iteration = (i+1) - num_taken
                        taken_this_iter = min(num_needed_this_iteration, np.sum(data_masks[i]))
                        num_taken += taken_this_iter
                        inds = np.argsort(qualities[data_masks[i]])
                        for j in range(taken_this_iter):
                            reg = (regions[data_masks[i]][inds[j]]).tolist()
                            img = image_names[data_masks[i]][inds[j]]
                            print("adding region {} to image {}".format(reg, img))

                            annotations[img]["training_regions"].append(reg)
                        

                    # r0_lowest_quality_index = np.argmin(qualities[q0_data_mask])
                    # r0_reg = regions[q0_data_mask][r0_lowest_quality_index]
                    # r0_img = image_names[q0_data_mask][r0_lowest_quality_index]

                    # r1_lowest_quality_index = np.argmin(qualities[q1_data_mask])
                    # r1_reg = regions[q1_data_mask][r1_lowest_quality_index]
                    # r1_img = image_names[q1_data_mask][r1_lowest_quality_index]

                    # r2_lowest_quality_index = np.argmin(qualities[q2_data_mask])
                    # r2_reg = regions[q2_data_mask][r2_lowest_quality_index]
                    # r2_img = image_names[q2_data_mask][r2_lowest_quality_index]

                    # r3_lowest_quality_index = np.argmin(qualities[q3_data_mask])
                    # r3_reg = regions[q3_data_mask][r3_lowest_quality_index]
                    # r3_img = image_names[q3_data_mask][r3_lowest_quality_index]

                    # annotations[r0_img].append(r0_reg)
                    # annotations[r1_img].append(r1_reg)
                    # annotations[r2_img].append(r2_reg)
                    # annotations[r3_img].append(r3_reg)


                elif method["method_name"] == "img_split":

                    quality_tuples.sort(key=lambda x: x[0])
                    inds = []
                    if method["num_low"] > 0:
                        for i in range(method["num_low"]):
                            inds.append(i)

                    if method["num_mid"] > 0:
                        start_pt = int(len(quality_tuples) / 2) - int(method["num_mid"] / 2)
                        for i in range(method["num_mid"]):
                            inds.append(start_pt + i)
                    if method["num_high"] > 0:
                        for i in range(method["num_high"]):
                            inds.append(len(quality_tuples) - i)
                    # for x in quality_tuples[:num_regions_per_image]:
                    for ind in inds:
                        x = quality_tuples[ind]
                        image_name = x[1]
                        new_region = x[2]
                        annotations[image_name]["training_regions"].append(new_region)


                    # chosen_images = [x[1] for x in quality_tuples[:num_regions_per_image]]

            else:
                raise RuntimeError("Specified number of regions is not supported")

        elif method["method_name"] == "regions_match_image_anno_count" or method["method_name"] == "low_quality_regions_match_image_anno_count":

            num_annotations_per_region = method["num_annotations_per_region"]
            matched_image_set = method["matched_image_set"]

            matched_image_set_dir = os.path.join("usr", "data", matched_image_set["username"], "image_sets",
                matched_image_set["farm_name"], matched_image_set["field_name"], matched_image_set["mission_date"])

            # last_matched_result_dir = get_most_recent_result_dir(matched_image_set)
            # matched_annotations_path = os.path.join(results_dir, "annotations.json")
            # matched_annotations = annotation_utils.load_annotations(matched_annotations_path)
            # num_matched_annotations = []
            # for image_name in matched_annotations.keys():
            #     if len(matched_annotations[image_name]["training_regions"]) > 0:
            #         num_matched_annotations += matched_annotations[image_name]["boxes"].shape[0]

            result_pairs = []
            results_dir = os.path.join(matched_image_set_dir, "model", "results")
            for result_dir in glob.glob(os.path.join(results_dir, "*")):
                request_path = os.path.join(result_dir, "request.json")
                request = json_io.load_json(request_path)
                end_time = request["end_time"]
                result_pairs.append((result_dir, end_time))

            result_pairs.sort(key=lambda x: x[1])

            matched_images = []
            matched_annotation_counts = []
            for result_pair in result_pairs:
                result_dir = result_pair[0]
                matched_annotations_path = os.path.join(result_dir, "annotations.json")
                matched_annotations = annotation_utils.load_annotations(matched_annotations_path)
                for image_name in matched_annotations:
                    if len(matched_annotations[image_name]["training_regions"]) > 0:
                        if image_name not in matched_images:
                            matched_images.append(image_name)
                            matched_annotation_counts.append(matched_annotations[image_name]["boxes"].shape[0])

            # print("Num annotations in matched image set: {}".format(num_matched_annotations))

            results_dir = os.path.join(image_set_dir, "model", "results")
            current_iteration_number = len(glob.glob(os.path.join(results_dir, "*")))
            current_index = current_iteration_number - 1


            print("Current iteration number: {}. Current index: {}".format(current_iteration_number, current_index))

            # cur_count = 0
            # for image_name in annotations.keys():
            #     if len(annotations[image_name]["training_regions"]) > 0:
            #         cur_count += box_utils.get_fully_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size


            # print("Matched annotation counts: {}".format(matched_annotation_counts))
            # print("Current count: {}".format(cur_count))

            # index = 0
            # while np.sum(np.array(matched_annotation_counts)[:index]) != cur_count:
            #     index += 1

            


            num_annotations_to_add = matched_annotation_counts[current_index]

            print("Number of annotations to add on this iteration: {}".format(num_annotations_to_add))

            # print("The most recently added image from the other method was {}.".format(matched_images[-1]))
            # print("This image contains {} annotations.".format(matched_annotation_counts[-1]))

            # image_names = list(annotations.keys())
            # random.shuffle(image_names)


            num_remaining = num_annotations_to_add
            # index = 0
            if method["method_name"] == "regions_match_image_anno_count":
                while num_remaining > 0:
                    num_to_add = min(num_remaining, num_annotations_per_region)
                    # if index > len(image_names):
                    #     raise RuntimeError("Ran out of choices")
                    image_names = list(annotations.keys())
                    random.shuffle(image_names)

                    image_name = image_names[0] #[index] #random.choice(annotations.keys())
                    image_h = metadata["images"][image_name]["height_px"]
                    image_w = metadata["images"][image_name]["width_px"]
                    print("Adding region with {} annotations to  image {}.".format(num_to_add, image_name))
                    new_region = random_region(annotations, num_to_add, image_name, image_h, image_w)
                    if new_region is not None:
                        
                        num_actually_added = box_utils.get_contained_inds(annotations[image_name]["boxes"], [new_region]).size
                        
                        if num_remaining <= num_annotations_per_region and num_actually_added != num_to_add:
                            pass
                        else:
                            print("Actually added {} annotations".format(num_actually_added))
                            annotations[image_name]["training_regions"].append(new_region)
                            # num_added += 1
                            num_remaining = num_remaining - num_actually_added #num_to_add
                            # index += 1


            else:
                quality_tuples = []
                for image_name in annotations.keys():
                    quality = get_confidence_quality(np.array(predictions[image_name]["scores"]))
                    quality_tuples.append((quality, image_name))
                quality_tuples.sort(key=lambda x: x[0])
                image_names = [x[1] for x in quality_tuples]

                index = 0
                while num_remaining > 0:
                    num_to_add = min(num_remaining, num_annotations_per_region)
                    if index > len(image_names):
                        raise RuntimeError("Ran out of choices")
                    # image_names = list(annotations.keys())
                    # random.shuffle(image_names)

                    image_name = image_names[index] #[index] #random.choice(annotations.keys())
                    image_h = metadata["images"][image_name]["height_px"]
                    image_w = metadata["images"][image_name]["width_px"]
                    print("Adding region with {} annotations to  image {}.".format(num_to_add, image_name))
                    new_region = random_region(annotations, num_to_add, image_name, image_h, image_w)
                    if new_region is not None:
                        
                        num_actually_added = box_utils.get_contained_inds(annotations[image_name]["boxes"], [new_region]).size
                        
                        if num_remaining <= num_annotations_per_region and num_actually_added != num_to_add:
                            pass
                        else:
                            print("Actually added {} annotations".format(num_actually_added))
                            annotations[image_name]["training_regions"].append(new_region)
                            # num_added += 1
                            num_remaining = num_remaining - num_actually_added #num_to_add
                            # index += 1
                    index += 1

            print("Done adding annotations.")

            # last_matched_result_dir = get_most_recent_result_dir(matched_image_set)
            
            # num_annotations_per_region = method["num_annotations_per_region"]
            # # for i in range(num_regions):
            # image_names = list(annotations.keys())
            # random.shuffle(image_names)
            # num_added = 0
            # index = 0
            # while num_added < num_regions:
            #     if index > len(image_names):
            #         raise RuntimeError("Ran out of choices")
            #     image_name = image_names[index] #random.choice(annotations.keys())
            #     image_h = metadata["images"][image_name]["height_px"]
            #     image_w = metadata["images"][image_name]["width_px"]
            #     new_region = random_region(annotations, num_annotations_per_region, image_name, image_h, image_w)




        else:

            num_regions = method["num_regions"]
            num_annotations_per_region = method["num_annotations_per_region"]

            quality_tuples = []
            for image_name in predictions.keys():

                contained_inds = box_utils.get_contained_inds(np.array(predictions[image_name]["boxes"]), annotations[image_name]["training_regions"])
                mask = np.full(np.array(predictions[image_name]["scores"]).size, True)
                mask[contained_inds] = False
                considered_scores = np.array(predictions[image_name]["scores"])[mask]
                quality = get_confidence_quality(considered_scores)
                quality_tuples.append((quality, image_name))
            quality_tuples.sort(key=lambda x: x[0])



            # for i in range(num_regions):
            num_added = 0
            index = 0
            while num_added < num_regions:
                if index > len(quality_tuples):
                    raise RuntimeError("Ran out of choices")
                image_name = quality_tuples[index][1]
                image_h = metadata["images"][image_name]["height_px"]
                image_w = metadata["images"][image_name]["width_px"]

                if method["method_name"] == "sel_img_rand_reg":
                    new_region = random_region(annotations, num_annotations_per_region, image_name, image_h, image_w)
                elif method["method_name"] == "sel_img_sel_reg":
                    new_region = confidence_guided_region(annotations, predictions, num_annotations_per_region, image_name, image_h, image_w)
                else:
                    raise RuntimeError("Unrecognized method name: {}".format(method["method_name"]))

                if new_region is not None:
                    annotations[image_name]["training_regions"].append(new_region)
                    num_added += 1
                index += 1

    else:
        raise RuntimeError("Unrecognized method name: {}".format(method["method_name"]))


    annotation_utils.save_annotations(annotations_path, annotations)



def test(methods):


    logger = logging.getLogger(__name__)

    # sch_ctx["switch_queue"] = LockQueue()
    # sch_ctx["auto_select_queue"] = LockQueue()
    # sch_ctx["prediction_queue"] = LockQueue()
    # sch_ctx["training_queue"] = LockQueue()
    # sch_ctx["baseline_queue"] = LockQueue()

    for method in methods:
        image_set = method["image_set"]
        # training_image_selection_method = method["training_image_selection_method"]

        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
        image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
        annotations = json_io.load_json(
                            os.path.join(image_set_dir, "annotations", "annotations.json")
        )
        metadata = json_io.load_json(
                        os.path.join(image_set_dir, "metadata", "metadata.json")
        )

        status = json_io.load_json(os.path.join(image_set_dir, "model", "status.json"))
        image_names = list(annotations.keys())
        regions = []
        for image_name in metadata["images"].keys():
            regions.append([[0, 0, 
            metadata["images"][image_name]["height_px"],
            metadata["images"][image_name]["width_px"]]])




        for _ in range(method["num_iterations"]): #num_iterations):

            iteration_number = len(glob.glob(os.path.join(image_set_dir, "model", "results", "*")))

            if iteration_number > 0:

                add_training_annotations(image_set, method)


                # annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
                # # annotations = json_io.load_json(annotations_path)
                # # candidates = []
                # # for image_name in annotations.keys():
                # #     if len(annotations[image_name]["test_regions"]) > 0:
                # #         candidates.append(image_name)
                # # chosen_image_name = random.choice(candidates)
                # # if training_image_selection_method == "predetermined" or training_image_selection_method == "sequential":
                # #     chosen_image_name = method["selection_order"][iteration_number-1]
                # # else:
                

                # chosen_image_names = select_training_images(image_set, training_image_selection_method, 5)
                # for chosen_image_name in chosen_image_names:
                #     print("Chose new training image: {}".format(chosen_image_name))
                #     annotations[chosen_image_name]["training_regions"].append(annotations[chosen_image_name]["test_regions"][0])
                #     annotations[chosen_image_name]["test_regions"] = []
                # json_io.save_json(annotations_path, annotations)

                # logger.info("Emitting {} to {}".format(image_set, server_url + "/add_request"))
                # requests.post(server_url + "/add_request", data=image_set, verify=False)

                server.sch_ctx["training_queue"].enqueue(image_set)
                    
                #     {
                #     "username": username,
                #     "farm_name": farm_name,
                #     "field_name": field_name,
                #     "mission_date": mission_date
                # })
                train_queue_size = server.sch_ctx["training_queue"].size()
                print("train_queue_size", train_queue_size)
                while train_queue_size > 0:
                    item = server.sch_ctx["training_queue"].dequeue()
                    print("running process_train")

                    # annotations = annotation_utils.load_annotations(annotations_path)

                    # num_training_regions = annotation_utils.get_num_training_regions(annotations)

                    # updated_patch_size = ep.update_model_patch_size(image_set_dir, annotations, ["training_regions"])
                    # update_applied = ep.update_training_patches(image_set_dir, annotations, updated_patch_size)

                    # if update_applied:
                    #     image_set_aux.update_training_tf_records(image_set_dir, annotations) #changed_training_image_names, annotations)
                    #     image_set_aux.reset_loss_record(image_set_dir)


                    re_enqueue = server.process_train(item)



                    if re_enqueue:
                        server.sch_ctx["training_queue"].enqueue(item)
                    train_queue_size = server.sch_ctx["training_queue"].size()

                print("FINE_TUNE_EVAL: finished training for this iteration")

            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": image_names,
                "regions": regions,
                "save_result": True,
                "results_name": status["model_name"] + "_fine_tune_" + str(iteration_number),
                "results_message": ""
            }

            request_path = os.path.join(image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(image_set)

    # create_eval_chart(methods)



    # training_finished, re_enqueue = yolov4_image_set_driver.train(sch_ctx, image_set_dir)


def run_methods(methods, org_image_set, num_replications):

    for i in range(num_replications):
        for method in methods:
            org_image_set_dir = os.path.join("usr", "data", org_image_set["username"], "image_sets",
                                        org_image_set["farm_name"], org_image_set["field_name"], org_image_set["mission_date"])

            method["image_set"] = {
                "username": org_image_set["username"],
                "farm_name": org_image_set["farm_name"] + ":" + method["method_label"] + ":rep_" + str(i),
                "field_name": org_image_set["field_name"] + ":" + method["method_label"] + ":rep_" + str(i),
                "mission_date": org_image_set["mission_date"]
            }

            method_image_set_dir = os.path.join("usr", "data", method["image_set"]["username"], "image_sets",
                                        method["image_set"]["farm_name"], method["image_set"]["field_name"], method["image_set"]["mission_date"])
            field_dir = os.path.join("usr", "data", method["image_set"]["username"], "image_sets",
                                        method["image_set"]["farm_name"], method["image_set"]["field_name"])
            # os.makedirs(field_dir, exist_ok=False)
            # shutil.copytree(org_image_set_dir, method_image_set_dir)

            test([method])

        # annotations_path = os.path.join(method_image_set_dir, "annotations", "annotations.json")
        # annotations = annotation_utils.load_annotations(annotations_path)

        # for image_name in annotations.keys():
        #     annotations[image_name]["test_regions"] = []
        #     annotations[image_name]["training_regions"] = []
        # annotation_utils.save_annotations(annotations_path, annotations)







def run():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


    logging.basicConfig(level=logging.INFO)

    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()


    # method_0 = {}
    # method_0["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "test_ft_1_rand",
    #     "field_name": "test_ft_1_rand",
    #     "mission_date": "2023-01-12" 
    # }
    # method_0["training_image_selection_method"] = "random"


    # method_1 = {}
    # method_1["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "test_ft_1",
    #     "field_name": "test_ft_1",
    #     "mission_date": "2023-01-12" 
    # }
    # method_1["training_image_selection_method"] = "lowest_quality"



    # method_2 = {}
    # method_2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "test_ft_1_best_qual",
    #     "field_name": "test_ft_1_best_qual",
    #     "mission_date": "2023-01-12" 
    # }
    # method_2["training_image_selection_method"] = "highest_quality"


    # method_0 = {}
    # method_0["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_rand",
    #     "field_name": "artificial_ft_1_rand",
    #     "mission_date": "2023-01-13" 
    # }
    # method_0["training_image_selection_method"] = "random"


    # method_1 = {}
    # method_1["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_lowq",
    #     "field_name": "artificial_ft_1_lowq",
    #     "mission_date": "2023-01-17" 
    # }
    # method_1["training_image_selection_method"] = "lowest_quality"

    # method_2 = {}
    # method_2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_r",
    #     "field_name": "artificial_ft_1_r",
    #     "mission_date": "2023-01-14" 
    # }
    # method_2["training_image_selection_method"] = "random"


    # method_3 = {}
    # method_3["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_dist",
    #     "field_name": "artificial_ft_1_dist",
    #     "mission_date": "2023-01-14" 
    # }
    # method_3["training_image_selection_method"] = "predetermined"
    # # method_3["selection_order"] = get_selected_order_for_image_set(method_3["image_set"])

    # # print("selection order:", method_3["selection_order"])
    # method_4 = {}
    # method_4["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_seq",
    #     "field_name": "artificial_ft_1_seq",
    #     "mission_date": "2023-01-13" 
    # }
    # image_set_dir = os.path.join("usr", "data", method_4["image_set"]["username"], "image_sets",
    #     method_4["image_set"]["farm_name"], method_4["image_set"]["field_name"], 
    #     method_4["image_set"]["mission_date"])
    # images_dir = os.path.join(image_set_dir, "images")
    # images = [os.path.basename(x)[:-4] for x in glob.glob(os.path.join(images_dir, "*"))]
    # sorted_images = natsorted(images, key=lambda y: y.lower())
    # method_4["selection_order"] = sorted_images
    # print("selection order:", method_4["selection_order"])

    # method_4["training_image_selection_method"] = "sequential"

    # # test([method_1], num_iterations=5)

    # method_5 = {}
    # method_5["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_t2",
    #     "field_name": "artificial_ft_1_t2",
    #     "mission_date": "2023-01-16" 
    # }
    # method_5["training_image_selection_method"] = "regions"

    # method_6 = {}
    # method_6["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_t3",
    #     "field_name": "artificial_ft_1_t3",
    #     "mission_date": "2023-01-16" 
    # }
    # method_6["training_image_selection_method"] = "regions_reset"

    # method_7 = {}
    # method_7["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "artificial_ft_1_t4",
    #     "field_name": "artificial_ft_1_t4",
    #     "mission_date": "2023-01-16" 
    # }
    # method_7["training_image_selection_method"] = "large_regions"
    # methods = [method_2]
    # test(methods, num_iterations=10)
    # methods = [method_0, method_1, method_2]
    # create_eval_chart([method_1, method_2, method_3, method_4], os.path.join("fine_tuning_charts", "artificial_ft_1.svg")) #[method_2, method_3])

    # create_boxplot_comparison([method_1, method_3], os.path.join("fine_tuning_charts", "boxplots", "artificial_ft_1.svg"))
    
    # create_eval_chart_annotations([method_2, method_5], "accuracy", os.path.join("fine_tuning_charts", "artificial_ft_1_annotations_accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations([method_2, method_5], "dic", os.path.join("fine_tuning_charts", "artificial_ft_1_annotations_dic.svg"))
    # create_eval_chart_annotations([method_2, method_5], "percent_count_error", os.path.join("fine_tuning_charts", "artificial_ft_1_annotations_percent_count_error.svg"))

    # method = {}
    # method["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "bl_k_rand1_no_base",
    #     "field_name": "bl_k_rand1_no_base",
    #     "mission_date": "2023-01-18" 
    # }
    # method["training_image_selection_method"] = "random_no_base"


    # method2 = {}
    # method2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "bl_k_rand1",
    #     "field_name": "bl_k_rand1",
    #     "mission_date": "2023-01-18" 
    # }
    # method2["training_image_selection_method"] = "random"

    # method3 = {}
    # method3["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "bl_k_region",
    #     "field_name": "bl_k_region",
    #     "mission_date": "2023-01-18" 
    # }
    # method3["training_image_selection_method"] = "regions"

    # create_eval_chart_annotations([method, method2, method3], "accuracy", os.path.join("fine_tuning_charts", "bl_k_accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations([method, method2, method3], "dic", os.path.join("fine_tuning_charts", "bl_k_dic.svg"))
    # create_eval_chart_annotations([method, method2, method3], "percent_count_error", os.path.join("fine_tuning_charts", "bl_k_percent_count_error.svg"))

    # method = {}
    # method["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Blocks2022_rand",
    #     "field_name": "Kernen2_rand",
    #     "mission_date": "2022-06-08" 
    # }
    # method["training_image_selection_method"] = "random"

    # method2 = {}
    # method2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Blocks2022_reg",
    #     "field_name": "Kernen2_reg",
    #     "mission_date": "2022-06-08" 
    # }
    # method2["training_image_selection_method"] = "regions"

    # create_eval_chart_annotations([method, method2], "accuracy", os.path.join("fine_tuning_charts", "Blocks2022_Kernen2_2022-06-08_accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations([method, method2], "dic", os.path.join("fine_tuning_charts", "Blocks2022_Kernen2_2022-06-08_dic.svg"))
    # create_eval_chart_annotations([method, method2], "percent_count_error", os.path.join("fine_tuning_charts", "Blocks2022_Kernen2_2022-06-08_percent_count_error.svg"))
    
    # method = {}
    # method["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Blocks2022_rand_all_stages",
    #     "field_name": "Kernen2_rand_all_stages",
    #     "mission_date": "2022-06-08" 
    # }
    # method["training_image_selection_method"] = "random"

    # method2 = {}
    # method2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Blocks2022_reg_all_stages",
    #     "field_name": "Kernen2_reg_all_stages",
    #     "mission_date": "2022-06-08" 
    # }
    # method2["training_image_selection_method"] = "regions"

    # create_eval_chart_annotations([method, method2], "accuracy", os.path.join("fine_tuning_charts", "Blocks2022_Kernen2_2022-06-08_all_stages_accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations([method, method2], "dic", os.path.join("fine_tuning_charts", "Blocks2022_Kernen2_2022-06-08_all_stages_dic.svg"))
    # create_eval_chart_annotations([method, method2], "percent_count_error", os.path.join("fine_tuning_charts", "Blocks2022_Kernen2_2022-06-08_all_stages_percent_count_error.svg"))


    # method = {}
    # method["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Arvalis_3_rand",
    #     "field_name": "Arvalis_3_rand",
    #     "mission_date": "2023-01-19" 
    # }
    # method["training_image_selection_method"] = "random"

    # method2 = {}
    # method2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Arvalis_3_low_conf",
    #     "field_name": "Arvalis_3_low_conf",
    #     "mission_date": "2023-01-19" 
    # }
    # method2["training_image_selection_method"] = "lowest_quality"
    # method3 = {}
    # method3["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Arvalis_3_hi_conf",
    #     "field_name": "Arvalis_3_hi_conf",
    #     "mission_date": "2023-01-19" 
    # }
    # method3["training_image_selection_method"] = "highest_quality"


    # # test([method, method2, method3], num_iterations=3)

    # create_eval_chart_annotations([method, method2, method3], "accuracy", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis3", "accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations([method, method2, method3], "dic", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis3", "dic.svg"))
    # create_eval_chart_annotations([method, method2, method3], "percent_count_error", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis3", "percent_count_error.svg"))


    # method = {}
    # method["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Arvalis_10_rand",
    #     "field_name": "Arvalis_10_rand",
    #     "mission_date": "2023-01-20" 
    # }
    # method["training_image_selection_method"] = "random"

    # method2 = {}
    # method2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Arvalis_10_low_conf",
    #     "field_name": "Arvalis_10_low_conf",
    #     "mission_date": "2023-01-20" 
    # }
    # method2["training_image_selection_method"] = "lowest_quality"



    # # test([method, method2], num_iterations=2)

    # create_eval_chart_annotations([method, method2], "accuracy", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis10", "accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations([method, method2], "dic", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis10", "dic.svg"))
    # create_eval_chart_annotations([method, method2], "percent_count_error", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis10", "percent_count_error.svg"))

    # method = {}
    # method["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "SaskatoonEast_rand",
    #     "field_name": "Stevenson5SW_rand",
    #     "mission_date": "2022-06-13" 
    # }
    # method["training_image_selection_method"] = "random"

    # method2 = {}
    # method2["image_set"] = {
    #     "username": "erik",
    #     "farm_name": "Arvalis_10_low_conf",
    #     "field_name": "Arvalis_10_low_conf",
    #     "mission_date": "2023-01-20" 
    # }
    # method2["training_image_selection_method"] = "lowest_quality"
    # test([method], num_iterations=2)

    # create_eval_chart_annotations([method, method2], "accuracy", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis10", "accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations([method, method2], "dic", os.path.join("fine_tuning_charts", "comparisons", "Arvalis1", "Arvalis10", "dic.svg"))
    # creat


    # image_based_methods = ["rand_img", "sel_img"]
    # region_based_methods = ["rand_img_rand_reg", "sel_img_rand_reg", "sel_img_sel_reg"]

    org_image_set = {
        "username": "erik",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9S",
        "mission_date": "2022-06-07"
    }

    # org_image_set = {
    #     "username": "erik",
    #     "farm_name": "Blocks2022",
    #     "field_name": "Kernen4",
    #     "mission_date": "2022-06-08"
    # }

    method1 = {}
    method1["method_name"] = "rand_img"
    method1["method_label"] = "rand_img"
    method1["num_iterations"] = 1
    method1["num_images"] = 1

    method2 = {}
    method2["method_name"] = "sel_img"
    method2["method_label"] = "sel_img"
    method2["num_iterations"] = 1
    method2["num_images"] = 1

    method3 = {}
    method3["method_name"] = "rand_img_rand_reg"
    method3["method_label"] = "rand_img_rand_reg"
    method3["num_iterations"] = 5
    method3["num_regions"] = 5
    method3["num_annotations_per_region"] = 10

    method4 = {}
    method4["method_name"] = "sel_img_rand_reg"
    method4["method_label"] = "sel_img_rand_reg"
    method4["num_iterations"] = 5
    method4["num_regions"] = 5
    method4["num_annotations_per_region"] = 10

    method5 = {}
    method5["method_name"] = "sel_img_sel_reg"
    method5["method_label"] = "sel_img_sel_reg"
    method5["num_iterations"] = 5
    method5["num_regions"] = 5
    method5["num_annotations_per_region"] = 10    

    method6 = {}
    method6["method_name"] = "sel_img_sel_reg"
    method6["method_label"] = "sel_img_sel_reg_one_reg_per_iter"
    method6["num_iterations"] = 25
    method6["num_regions"] = 1
    method6["num_annotations_per_region"] = 10

    method7 = {}
    method7["method_name"] = "img_split"
    method7["method_label"] = "img_split_3_low_1_mid"
    method7["num_low"] = 3
    method7["num_mid"] = 1
    method7["num_high"] = 0
    method7["num_iterations"] = 5
    method7["num_regions_per_image"] = 4

    method8 = {}
    method8["method_name"] = "img_split"
    method8["method_label"] = "img_split_4_low"
    method8["num_low"] = 4
    method8["num_mid"] = 0
    method8["num_high"] = 0
    method8["num_iterations"] = 10
    method8["num_regions_per_image"] = 4

    method9 = {}
    method9["method_name"] = "img_split"
    method9["method_label"] = "img_split_4_low_min_20"
    method9["num_low"] = 4
    method9["num_mid"] = 0
    method9["num_high"] = 0
    method9["num_iterations"] = 5
    method9["num_regions_per_image"] = 4


    method10 = {}
    method10["method_name"] = "rand_img"
    method10["method_label"] = "rand_img_random_initial"
    method10["num_iterations"] = 1
    method10["num_images"] = 1

    method11 = {}
    method11["method_name"] = "sel_img"
    method11["method_label"] = "sel_img_random_initial"
    method11["num_iterations"] = 1
    method11["num_images"] = 1

    method12 = {}
    method12["method_name"] = "quartile"
    method12["method_label"] = "quartile_random_initial"
    method12["num_iterations"] = 1
    method12["num_regions_per_image"] = 4


    method13 = {}
    method13["method_name"] = "rand_img"
    method13["method_label"] = "rand_img_random_initial_fixed_epoch_num"
    method13["num_iterations"] = 5
    method13["num_images"] = 1

    method14 = {}
    method14["method_name"] = "sel_img"
    method14["method_label"] = "sel_img_random_initial_fixed_epoch_num"
    method14["num_iterations"] = 5
    method14["num_images"] = 1

    method15 = {}
    method15["method_name"] = "quartile"
    method15["method_label"] = "quartile_random_initial_fixed_epoch_num"
    method15["num_iterations"] = 5
    method15["num_regions_per_image"] = 4

    method16 = {}
    method16["method_name"] = "regions_match_image_anno_count"
    method16["method_label"] = "regions_match_image_anno_count_random_initial"
    method16["matched_image_set"] = {
        "username": "erik",
        "farm_name": "BlaineLake:rand_img_random_initial:rep_0",
        "field_name": "Serhienko9S:rand_img_random_initial:rep_0",
        "mission_date": "2022-06-07"
    }
    method16["num_annotations_per_region"] = 10
    method16["num_iterations"] = 2

    method17 = {}
    method17["method_name"] = "low_quality_regions_match_image_anno_count"
    method17["method_label"] = "low_quality_regions_match_image_anno_count_random_initial"
    method17["matched_image_set"] = {
        "username": "erik",
        "farm_name": "BlaineLake:rand_img_random_initial:rep_0",
        "field_name": "Serhienko9S:rand_img_random_initial:rep_0",
        "mission_date": "2022-06-07"
    }
    method17["num_annotations_per_region"] = 10
    method17["num_iterations"] = 7
    
    method18 = {}
    method18["method_name"] = "sel_worst_img"
    method18["method_label"] = "sel_worst_img"
    method18["num_iterations"] = 1
    method18["num_images"] = 1


    method19 = {}
    method19["method_name"] = "rand_img_rand_reg"
    method19["method_label"] = "rand_img_rand_reg_v2"
    method19["num_iterations"] = 5
    method19["num_regions"] = 5
    method19["num_annotations_per_region"] = 10

    # methods = [method1, method2, method3, method4, method5] #[method1, method2, method3, method4, method5] #, method5]
    
    methods = [method1] #[method1, method2] #, method5, method7, method8, method9]

    num_replications = 4
    run_methods(methods, org_image_set, num_replications)


    # create_eval_chart_annotations(org_image_set, methods, "accuracy", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "accuracy.svg")) #[method_2, method_3])
    # create_eval_chart_annotations(org_image_set, methods, "dic", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "dic.svg"))
    # create_eval_chart_annotations(org_image_set, methods, "percent_count_error", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "percent_count_error.svg"))
    # create_eval_chart_annotations(org_image_set, methods, "global_accuracy", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "global_accuracy.svg")) #[method_2, method_3])
    # create_boxplot_comparison(org_image_set, [method1, method2, method5, method7, method9], "dic", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "boxplots"), xpositions="num_annotations")
    # create_boxplot_comparison(org_image_set, [method1, method2, method5, method7, method9], "dic", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "boxplots"), xpositions="num_iterations")
    # create_boxplot_comparison(org_image_set, [method10, method16], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "boxplots"), xpositions="num_annotations")
    # create_boxplot_comparison(org_image_set, [method10, method16], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "boxplots"), xpositions="num_iterations")
    # create_eval_chart_annotations_2(org_image_set, [method10, method11, method12], "accuracy", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "min_needed_min_accuracy.svg")) #[method_2, method_3])
    # create_thinline_comparison(org_image_set, [method1, method5], "dic", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "thinline"), xpositions="num_annotations")
    # create_thinline_comparison(org_image_set, [method1, method5], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "thinline"), xpositions="num_annotations")
    # create_thinline_comparison(org_image_set, [method1, method5], "dic", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "thinline"), xpositions="num_iterations")
    # create_thinline_comparison(org_image_set, [method1, method5], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "thinline"), xpositions="num_iterations")

    # # global_accuracy_plot(org_image_set, [method1, method2], 1,  os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "global_accuracy_plot.svg"))
    # create_global_comparison(org_image_set, [method1, method5], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "global"), xpositions="num_annotations")

    image_set_str = org_image_set["farm_name"] + ":" + org_image_set["field_name"] + ":" + org_image_set["mission_date"]
    # create_thinline_comparison(org_image_set, [method1, method9], "abs_dic", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", image_set_str, "thinline"), xpositions="num_annotations", include_mean_line=False)
    # create_thinline_comparison(org_image_set, [method1, method2], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", image_set_str, "thinline"), xpositions="num_annotations")
    # create_thinline_comparison(org_image_set, [method1, method9], "abs_dic", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", image_set_str, "thinline"), xpositions="num_iterations", include_mean_line=False)
    # create_thinline_comparison(org_image_set, [method1, method2], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", image_set_str, "thinline"), xpositions="num_iterations")

    # create_global_comparison(org_image_set, [method1, method2], "accuracy", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", image_set_str, "global"), xpositions="num_annotations")
    # create_global_comparison(org_image_set, [method1, method2], "AP (IoU=.50)", 4, os.path.join("fine_tuning_charts", "comparisons", "all_stages", image_set_str, "global"), xpositions="num_annotations")


if __name__ == "__main__":
    run()
