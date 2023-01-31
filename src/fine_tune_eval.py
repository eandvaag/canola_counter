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
from fine_tune_dist_select import get_selected_order_for_image_set

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
                chosen_boxes = contained_boxes[sorted_inds[:10]]
                chosen_boxes = np.concatenate([chosen_boxes, np.expand_dims(np.tile(start_pt, 2), axis=0)])

                final_region = [
                    int(np.min(chosen_boxes[:,0])),
                    int(np.min(chosen_boxes[:,1])),
                    int(np.max(chosen_boxes[:,2])),
                    int(np.max(chosen_boxes[:,3]))
                ]

                
                for i in range(4):
                    if i == 0 or i == 1:
                        sign = -1
                    else:
                        sign = 1
                    save_coord = final_region[i]
                    pad = random.randint(10, 20)
                    final_region[i] = final_region[i] + (sign) * pad
                    apply_pad = True
                    for training_region in annotations[image_name]["training_regions"]:
                        if box_utils.get_intersection_rect(final_region, training_region)[0]:
                            apply_pad = False
                    for test_region in annotations[image_name]["test_regions"]:
                        if box_utils.get_intersection_rect(final_region, test_region)[0]:
                            apply_pad = False
                    if i == 0 and final_region[i] < 0:
                        apply_pad = False
                    if i == 1 and final_region[i] < 0:
                        apply_pad = False
                    if i == 2 and final_region[i] > image_h:
                        apply_pad = False
                    if i == 3 and final_region[i] > image_w:
                        apply_pad = False

                    if not apply_pad:
                        final_region[i] = save_coord



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

    random.shuffle(start_pts)
    start_pts = np.array(start_pts)
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
        
        ax.plot(chart_entry["num_annotations"], chart_entry["min_vals"], color=color, marker="x", linestyle="dashed", alpha=0.9) #, alpha=0.75, linestyle='dashed')
        ax.plot(chart_entry["num_annotations"], chart_entry["max_vals"], color=color, marker="x", linestyle="dashed", alpha=0.9) #, linestyle='dashed')
    
    ax.set_ylabel("Mean " + metric.capitalize()) # Accuracy")
    if metric == "accuracy":
        ax.set_ylim((0, 1))


    ax.set_xlabel("Number of Annotations Used For Fine-Tuning")

    ax.legend()
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    fig.savefig(out_path) #"artificial_ft_1_fine_tuning_method_comparison.svg")


def create_eval_chart(methods, out_path):
    # dfs = {}
    excluded = []

    # for method in methods:
    #     image_set = method["image_set"]
    #     training_image_selection_method = method["training_image_selection_method"]
    #     image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
    #                                 image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

    #     results_dir = os.path.join(image_set_dir, "model", "results")
    #     result_dirs = glob.glob(os.path.join(results_dir, "*"))
    #     result_tuples = []
    #     for result_dir in result_dirs:
    #         annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
    #         num_training_images = 0
    #         for image_name in annotations.keys():
    #             if len(annotations[image_name]["training_regions"]) > 0:
    #                 num_training_images += 1
    #         result_tuples.append((num_training_images, result_dir))
    #     result_tuples.sort(key=lambda x: x[0])
    #     last_result_dir = result_tuples[-1][1]

    #     df = pd.read_excel(os.path.join(last_result_dir, "metrics.xlsx"))

    #     excluded.extend(df.index[df["Image Is Fully Annotated"] == "yes: for fine-tuning"].tolist())
    #     # dfs[training_image_selection_method] = df
    # print("{} images are excluded because they were used for testing.".format((np.unique(excluded)).size))
    


    for method in methods:
        image_set = method["image_set"]
        training_image_selection_method = method["training_image_selection_method"]
        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                    image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

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

        method["mean_accuracies"] = []
        for result_tuple in result_tuples:
            df = pd.read_excel(os.path.join(result_tuple[1], "metrics.xlsx"))

            # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
            # subset_df = df.iloc[included_rows]

            # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])
            mean_accuracy = np.mean(df["Accuracy (IoU=.50, conf>.50)"])

            method["mean_accuracies"].append(mean_accuracy)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    for method in methods:
        print(method)
        training_image_selection_method = method["training_image_selection_method"]
        ax.plot(np.arange(0, len(method["mean_accuracies"])), method["mean_accuracies"], label=training_image_selection_method)
    
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim((0, 1))

    ax.set_xlabel("Number of Images Used For Fine-Tuning")

    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path) #"artificial_ft_1_fine_tuning_method_comparison.svg")



def create_boxplot_comparison(methods, out_path):

    # methods = [method_1, method_2]
    # excluded = []

    # for method in methods:
    #     image_set = method["image_set"]
    #     training_image_selection_method = method["training_image_selection_method"]
    #     image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
    #                                 image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

    #     results_dir = os.path.join(image_set_dir, "model", "results")
    #     result_dirs = glob.glob(os.path.join(results_dir, "*"))
    #     result_tuples = []
    #     for result_dir in result_dirs:
    #         annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))
    #         num_training_images = 0
    #         for image_name in annotations.keys():
    #             if len(annotations[image_name]["training_regions"]) > 0:
    #                 num_training_images += 1
    #         result_tuples.append((num_training_images, result_dir))
    #     result_tuples.sort(key=lambda x: x[0])
    #     last_result_dir = result_tuples[-1][1]

    #     df = pd.read_excel(os.path.join(last_result_dir, "metrics.xlsx"))

    #     excluded.extend(df.index[df["Image Is Fully Annotated"] == "yes: for fine-tuning"].tolist())
    #     # dfs[training_image_selection_method] = df
    # print("{} images are excluded because they were used for testing.".format((np.unique(excluded)).size))
    
    for method in methods:
        method["accuracies"] = []
        image_set = method["image_set"]
        training_image_selection_method = method["training_image_selection_method"]
        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                    image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

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

        # method["mean_accuracies"] = []
        for result_tuple in result_tuples:
            df = pd.read_excel(os.path.join(result_tuple[1], "metrics.xlsx"))

            # included_rows = [x for x in range(0, len(df[df.keys()[0]])) if x not in excluded]
            # subset_df = df.iloc[included_rows]

            # mean_accuracy = np.mean(subset_df["Accuracy (IoU=.50, conf>.50)"])

            method["accuracies"].append(df["Accuracy (IoU=.50, conf>.50)"])

    fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(111)

    method_0_plot = plt.boxplot(methods[0]["accuracies"], positions=np.array(np.arange(len(methods[0]["accuracies"]))) * 2.0 - 0.3, widths=0.2, whis=(0, 100))
    method_1_plot =   plt.boxplot(methods[1]["accuracies"], positions=np.array(np.arange(len(methods[1]["accuracies"]))) * 2.0 + 0.3, widths=0.2, whis=(0, 100))

    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
            
        # use plot function to draw a small line to name the legend.
        plt.plot([], c=color_code, label=label)
        plt.legend()
 
 
    # setting colors for each groups
    define_box_properties(method_0_plot, '#D7191C', methods[0]["training_image_selection_method"])
    define_box_properties(method_1_plot, '#2C7BB6', methods[1]["training_image_selection_method"])
    

    ticks = np.arange(0, len(methods[0]["accuracies"]))
    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    
    # set the limit for x axis
    plt.xlim(-2, len(ticks)*2)
    
    # set the limit for y axis
    plt.ylim(0, 1.0)
    
    # set the title
    #plt.title('Grouped boxplot using matplotlib')

    plt.savefig(out_path)

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

    image_based_methods = ["rand_img", "sel_img"]
    region_based_methods = ["rand_img_rand_reg", "sel_img_rand_reg", "sel_img_sel_reg"]

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
        elif method["method_name"] == "sel_img":
            quality_tuples = []
            for image_name in candidates:
                quality = get_confidence_quality(np.array(predictions[image_name]["scores"]))
                quality_tuples.append((quality, image_name))
            quality_tuples.sort(key=lambda x: x[0])
            chosen_images = [x[1] for x in quality_tuples[:num_images]]

        for image_name in chosen_images:
            annotations[image_name]["training_regions"].append([
                0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]
            ])


    elif method["method_name"] in region_based_methods:
    

        num_regions = method["num_regions"]
        num_annotations_per_region = method["num_annotations_per_region"]
        if method["method_name"] == "rand_img_rand_reg":
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

        else:
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

    method1 = {}
    method1["method_name"] = "rand_img"
    method1["method_label"] = "rand_img"
    method1["num_iterations"] = 4
    method1["num_images"] = 1

    method2 = {}
    method2["method_name"] = "sel_img"
    method2["method_label"] = "sel_img"
    method2["num_iterations"] = 4
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
    method5["num_iterations"] = 2
    method5["num_regions"] = 5
    method5["num_annotations_per_region"] = 10    

    method6 = {}
    method6["method_name"] = "sel_img_sel_reg"
    method6["method_label"] = "sel_img_sel_reg_one_reg_per_iter"
    method6["num_iterations"] = 25
    method6["num_regions"] = 1
    method6["num_annotations_per_region"] = 10

    methods = [method1, method2, method3, method4, method5, method6] #[method1, method2, method3, method4, method5] #, method5]
    # methods = [method2, method5]

    num_replications = 4
    # run_methods(methods, org_image_set, num_replications)


    create_eval_chart_annotations(org_image_set, methods, "accuracy", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "accuracy.svg")) #[method_2, method_3])
    create_eval_chart_annotations(org_image_set, methods, "dic", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "dic.svg"))
    create_eval_chart_annotations(org_image_set, methods, "percent_count_error", num_replications, os.path.join("fine_tuning_charts", "comparisons", "all_stages", "BlaineLake:Serhienko9S:2022-06-07", "percent_count_error.svg"))


if __name__ == "__main__":
    run()
