import logging
import argparse
import tqdm
import os
import time
import numpy as np
from mean_average_precision import MetricBuilder
import matplotlib.pyplot as plt

import pandas as pd
import pandas.io.formats.excel
from natsort import index_natsorted

#from styleframe import StyleFrame



from models.common import box_utils, annotation_utils


from io_utils import json_io, w3c_io



# def DiC(actual, pred):
#     return actual - pred

# def abs_DiC(actual, pred):
#     return np.abs(actual - pred)

# def pct_DiC(actual, pred):
#     return np.divide(abs_DiC(actual, pred), actual, out=np.zeros_like(actual, dtype=np.float64), where=actual!=0) * 100

# def nonzero_abs_DiC(actual, pred):
#     mask = actual != 0
#     return abs_DiC(actual[mask], pred[mask])

# def squared_DiC(actual, pred):
#     return (actual - pred) ** 2

# def r_squared(actual, pred):
#     SS_res = np.sum((actual - pred) ** 2)
#     SS_tot = np.sum((actual - np.mean(actual)) ** 2)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         res = 1 - (SS_res / SS_tot)
#     if np.isnan(res):
#         res = 0.0 
#     return float(res)



# def boxplot_data(data):

#     data = np.array(data)
#     if data.size == 0:
#         res = {
#             "range_min": 0,
#             "whisker_min": 0,
#             "q1": 0,
#             "q2": 0,
#             "q3": 0,
#             "whisker_max": 0,
#             "range_max": 0,
#             "outliers": []
#         }
#     else:
#         q1, q2, q3 = np.percentile(data, [25, 50, 75], interpolation="midpoint")
#         iqr = q3 - q1

#         res = {
#             "range_min": float(np.min(data)),
#             "whisker_min": q1 - 1.5 * iqr,
#             "q1": q1,
#             "q2": q2,
#             "q3": q3,
#             "whisker_max": q3 + 1.5 * iqr,
#             "range_max": float(np.max(data))
#         }

#         res["outliers"] = data[np.logical_or(data < res["whisker_min"],
#                                              data > res["whisker_max"])].tolist()

#     return res




# # def five_num_summary(data):
# #     return np.percentile(np.array(data), [0, 25, 50, 75, 100], interpolation='midpoint').tolist()

# # def five_num_summary_DiC(actual, pred):
# #     return five_num_summary(actual - pred)

# # def DiC_occurrences(actual, pred):
# #     a = (actual - pred)
# #     vals = np.arange(a.min(), a.max() + 1)
# #     occurrences, _ = np.histogram(a, bins=(a.max() - a.min() + 1))
# #     return {
# #             "vals": vals.tolist(),
# #             "occurrences": occurrences.tolist()
# #            }


# def collect_statistics(image_names, metrics, predictions, config, inference_times=None):


#     # if "metrics" not in predictions:
#     #     predictions["metrics"] = {}

#     # if "point" not in predictions["metrics"]:
#     #     predictions["metrics"]["point"] = {}

#     # if "boxplot" not in predictions["metrics"]:
#     #     predictions["metrics"]["boxplot"] = {}

#     # datasets = {
#     #     "training": image_set.training_dataset,
#     #     "validation": image_set.validation_dataset,
#     #     "test", image_set.test_dataset
#     # }
#     #dataset = image_set.all_dataset

#     point_metrics = metrics["point"]
#     boxplot_metrics = metrics["boxplot"]

#     if inference_times is not None:
#         total_inference_time = float(np.sum(inference_times))
#         point_metrics["Total Inference Time (s)"] = {}
#         point_metrics["Total Inference Time (s)"]["---"] = total_inference_time
#         point_metrics["Per Patch Inference Time (s)"] = {}
#         point_metrics["Per Patch Inference Time (s)"]["---"] = total_inference_time / len(inference_times)
#         point_metrics["Per Image Inference Time (s)"] = {}
#         point_metrics["Per Image Inference Time (s)"]["---"] = total_inference_time / len(predictions["image_predictions"])

#     #boxplot_metrics["Inference Times (Per Patch)"] = {}
#     #boxplot_metrics["Inference Times (Per Patch)"]["---"] = boxplot_data(inference_times)

#     # point_metrics["Mean Confidence Score"] = {}
#     # point_metrics["Max Confidence Score"] = {}
#     # point_metrics["Max Box Area"] = {}
#     # point_metrics["Min Box Area"] = {}
#     # point_metrics["Mean Box Area"] = {}

#     boxplot_metrics["Confidence"] = {}
#     boxplot_metrics["Box Area"] = {}

#     confidences = {k: [] for k in config["arch"]["class_map"].keys()}
#     boxes = {k: [] for k in config["arch"]["class_map"].keys()}
#     #for image in dataset.images:
#     for image_name in image_names: #predictions["image_predictions"].keys():
#         for cls_name in config["arch"]["class_map"].keys():
#             cls_confs = predictions["image_predictions"][image_name]["pred_class_scores"][cls_name] 
#             confidences[cls_name].extend(cls_confs)

#             cls_boxes = predictions["image_predictions"][image_name]["pred_class_boxes"][cls_name]
#             boxes[cls_name].extend(cls_boxes)

#     num_predictions = np.sum(len(confidences[k]) for k in confidences.keys())

#     # point_metrics["Mean Confidence Score"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Max Confidence Score"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Mean Box Area"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Min Box Area"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Max Box Area"]["Cross-Class Weighted Average"] = 0

#     for cls_name in config["arch"]["class_map"].keys():

#         if len(confidences[cls_name]) > 0:
#             mean_cls_conf = float(np.mean(confidences[cls_name]))
#             max_cls_conf = float(np.max(confidences[cls_name]))
#             box_areas = box_utils.box_areas_np(np.array(boxes[cls_name]))
#             mean_box_area = float(np.mean(box_areas))
#             min_box_area = float(np.min(box_areas))
#             max_box_area = float(np.max(box_areas))

#         else:
#             mean_cls_conf = 0
#             max_cls_conf = 0
#             box_areas = np.array([])
#             mean_box_area = 0
#             min_box_area = 0
#             max_box_area = 0


#         boxplot_metrics["Confidence"][cls_name] = boxplot_data(confidences[cls_name])
#         boxplot_metrics["Box Area"][cls_name] = boxplot_data(box_areas)

#         # point_metrics["Mean Confidence Score"][cls_name] = mean_cls_conf
#         # point_metrics["Max Confidence Score"][cls_name] = max_cls_conf
#         # point_metrics["Mean Box Area"][cls_name] = mean_box_area
#         # point_metrics["Max Box Area"][cls_name] = max_box_area
#         # point_metrics["Min Box Area"][cls_name] = min_box_area
#         # if num_predictions > 0:
#         #     point_metrics["Mean Confidence Score"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * mean_cls_conf
#         #     point_metrics["Max Confidence Score"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * max_cls_conf

#         #     point_metrics["Mean Box Area"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * mean_box_area
#         #     point_metrics["Min Box Area"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * min_box_area
#         #     point_metrics["Max Box Area"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * max_box_area


# def calculate_optimal_score_threshold(annotations, predictions, image_names):

#     # currently assumes only class is plant class


#     logger = logging.getLogger(__name__)    

#     optimal_thresh_val = None
#     optimal_mean_abs_diff = np.inf
#     prev_mean_abs_diff = np.inf
#     thresh_vals = np.arange(0.0, 1.0, 0.01)
#     for thresh_val in tqdm.tqdm(thresh_vals, desc="Calculating optimal threshold value"):
#         abs_diffs = []
#         for image_name in image_names:
#             num_annotations = annotations[image_name]["boxes"].shape[0]
#             num_predictions = (np.where(np.array(predictions["image_predictions"][image_name]["pred_scores"]) >= thresh_val)[0]).size
#             abs_diffs.append(abs(num_annotations - num_predictions))
#         mean_abs_diff = float(np.mean(abs_diffs))

#         if prev_mean_abs_diff < mean_abs_diff:
#             break

#         if mean_abs_diff <= optimal_mean_abs_diff:
#             optimal_mean_abs_diff = mean_abs_diff
#             optimal_thresh_val = thresh_val

#         prev_mean_abs_diff = mean_abs_diff

#     logger.info("Optimal score threshold for all images is: {}".format(optimal_thresh_val))

#     return optimal_thresh_val, optimal_mean_abs_diff



# def similarity_analysis(config, predictions):

    

#     distance_record_path = os.path.join(config["model_dir"], "patch_distances.json")

#     if not os.path.exists(distance_record_path):
#         return

#     distance_record = json_io.load_json(distance_record_path)

#     class_map = config["arch"]["class_map"]
#     reverse_class_map = config["arch"]["reverse_class_map"]


#     # annotated_patch_counts = {k: [] for k in class_map.keys()}
#     # pred_patch_counts = {k: [] for k in class_map.keys()} 

#     res = []
#     for patch_name, patch_pred in predictions["patch_predictions"].items():
#         if "patch_classes" in patch_pred:

#             patch_classes = patch_pred["patch_classes"]
#             unique, counts = np.unique(patch_classes, return_counts=True)
#             class_num_to_count = dict(zip(unique, counts))
#             cur_patch_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
#             for class_num in class_num_to_count.keys():
#                 cur_patch_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
#                 #cur_patch_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

#             pred_patch_classes = patch_pred["pred_classes"]
#             pred_unique, pred_counts = np.unique(pred_patch_classes, return_counts=True)
#             class_num_to_pred_count = dict(zip(pred_unique, pred_counts))
#             cur_patch_pred_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}


#             annotated_patch_plant_count = cur_patch_class_counts["plant"]
#             pred_patch_plant_count = cur_patch_pred_class_counts["plant"]
            
#             abs_diff = abs(annotated_patch_plant_count - pred_patch_plant_count)
#             diff = (annotated_patch_plant_count - pred_patch_plant_count)
#             distance = distance_record[patch_name + ".png"]
#             res.append({
#                 "abs_diff": abs_diff,
#                 "diff": diff,
#                 "distance": distance
#             })
#             # for class_name in class_map.keys(): #config.arch["class_map"].keys():
#             #     annotated_patch_counts[class_name].append(cur_patch_class_counts[class_name])
#             #     pred_patch_counts[class_name].append(cur_patch_pred_class_counts[class_name])

#     xs = [d["distance"] for d in res]
#     ys = [d["abs_diff"] for d in res]

#     out_path = os.path.join(config["model_dir"], "distance_versus_abs_count_diff.png")
#     plt.figure()
#     plt.scatter(xs, ys)
#     plt.xlabel("Euclidean Feature Distance")
#     plt.ylabel("Absolute difference in predicted count")
#     plt.savefig(out_path)

#     xs = [d["distance"] for d in res]
#     ys = [d["diff"] for d in res]

#     out_path = os.path.join(config["model_dir"], "distance_versus_count_diff.png")
#     plt.figure()
#     plt.scatter(xs, ys)
#     plt.xlabel("Euclidean Feature Distance")
#     plt.ylabel("Difference in predicted count")
#     plt.savefig(out_path)

def get_f1_score(annotated_boxes, pred_boxes, iou_thresh):

    if annotated_boxes.size == 0 or pred_boxes.size == 0:
        return 0

    iou_mat = box_utils.compute_iou_np(annotated_boxes, pred_boxes)
    overlap_mat = iou_mat >= iou_thresh
    num_true_positives = np.any(overlap_mat, axis=1).sum()
    num_false_positives = np.all(np.logical_not(overlap_mat), axis=0).sum()
    num_false_negatives = np.all(np.logical_not(overlap_mat), axis=1).sum()

    print("num_true_positives", num_true_positives)
    print("num_false_positives", num_false_positives)
    print("num_false_negatives", num_false_negatives)

    if num_true_positives == 0 and num_false_positives == 0:
        return 0

    if num_true_positives == 0 and num_false_negatives == 0:
        return 0
    precision = num_true_positives / (num_true_positives + num_false_positives)
    recall = num_true_positives / (num_true_positives + num_false_negatives)


    if precision == 0 and recall == 0:
        return 0

    f1_score = (2 * precision * recall) / (precision + recall)

    return float(f1_score)


def collect_image_set_metrics(image_set_dir, full_predictions, annotations): #, config):

    logger = logging.getLogger(__name__)
    logger.info("Collecting image set metrics")

    start_time = int(time.time())

    # num_classes = len(config["arch"]["class_map"].keys())

    # image_metrics = {}

    # metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    # metadata = json_io.load_json(metadata_path)
    # is_ortho = metadata["is_ortho"] == "yes"

    # if is_ortho:
    metrics = {
    "AP (IoU=.50:.05:.95)": {},
    "AP (IoU=.50)": {},
    "AP (IoU=.75)": {},
    "F1 Score (IoU=.50, conf>=.50)" : {},
    "F1 Score (IoU=.75, conf>=.50)" : {}
    }
    # else:
    #     metrics = {
    #         "AP (IoU=.50:.05:.95)": {},
    #         "AP (IoU=.50)": {},
    #         "AP (IoU=.75)": {},
    #         "F1 Score (IoU=.50, conf>=.50)" : {},
    #         "F1 Score (IoU=.75, conf>=.50)" : {}
    #     }

    for image_name in annotations.keys():

        print("collect_image_set_metrics", image_name)

        metrics["AP (IoU=.50:.05:.95)"][image_name] = {}
        metrics["AP (IoU=.50)"][image_name] = {}
        metrics["AP (IoU=.75)"][image_name] = {}
        metrics["F1 Score (IoU=.50, conf>=.50)"][image_name] = {}
        metrics["F1 Score (IoU=.75, conf>=.50)"][image_name] = {}

        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(full_predictions[image_name]["boxes"])
        pred_scores = np.array(full_predictions[image_name]["scores"])

        for region_key in ["training_regions", "test_regions"]:
            metrics["AP (IoU=.50:.05:.95)"][image_name][region_key] = []
            metrics["AP (IoU=.50)"][image_name][region_key] = []
            metrics["AP (IoU=.75)"][image_name][region_key] = []
            metrics["F1 Score (IoU=.50, conf>=.50)"][image_name][region_key] = []
            metrics["F1 Score (IoU=.75, conf>=.50)"][image_name][region_key] = []


            for region in annotations[image_name][region_key]:

                
                region_annotated_inds = box_utils.get_contained_inds(annotated_boxes, [region])
                region_annotated_boxes = annotated_boxes[region_annotated_inds]
                # region_annotated_classes = np.zeros(shape=(region_annotated_boxes.shape[0]))
                
                region_pred_inds = box_utils.get_contained_inds(pred_boxes, [region])
                region_pred_boxes = pred_boxes[region_pred_inds]
                region_pred_scores = pred_scores[region_pred_inds]
                
                # sel_region_pred_scores = region_pred_scores[region_pred_scores >= 0.50]
                sel_region_pred_boxes = region_pred_boxes[region_pred_scores >= 0.50]
                # region_pred_classes = np.zeros(shape=(region_pred_boxes.shape[0]))

                print("getting AP vals")
                AP_vals = get_AP_vals(region_annotated_boxes, region_pred_boxes, region_pred_scores)

                metrics["AP (IoU=.50:.05:.95)"][image_name][region_key].append(AP_vals["AP (IoU=.50:.05:.95)"])
                metrics["AP (IoU=.50)"][image_name][region_key].append(AP_vals["AP (IoU=.50)"])
                metrics["AP (IoU=.75)"][image_name][region_key].append(AP_vals["AP (IoU=.75)"])

                # pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
                #     region_pred_boxes, region_pred_classes, region_pred_scores,
                #     region_annotated_boxes, region_annotated_classes)

                # image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
                # image_metric_fn.add(pred_for_mAP, true_for_mAP)
                # # pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
                # coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


                # image_metrics[image_name]["Image PASCAL VOC mAP"] = float(pascal_voc_mAP) * 100
                # metrics["MS COCO mAP"][image_name][region_key].append(MS_COCO_mAP)

                print("getting F1 scores")
                f1_iou_050 = get_f1_score(region_annotated_boxes, sel_region_pred_boxes, iou_thresh=0.50)
                f1_iou_075 = get_f1_score(region_annotated_boxes, sel_region_pred_boxes, iou_thresh=0.75)
                # f1_iou_09 = get_f1_score(region_annotated_boxes, sel_region_pred_boxes, iou_thresh=0.9)
                metrics["F1 Score (IoU=.50, conf>=.50)"][image_name][region_key].append(f1_iou_050)
                metrics["F1 Score (IoU=.75, conf>=.50)"][image_name][region_key].append(f1_iou_075)

    end_time = int(time.time())
    elapsed_time = end_time - start_time
    print("Finished calculating metrics. Took {} seconds.".format(elapsed_time))

    return metrics


    # for image_name in tqdm.tqdm(image_predictions.keys(), desc="Collecting metrics"):

    #     image_abs_boxes = annotations[image_name]["boxes"]
    #     image_classes = annotations[image_name]["classes"]
    #     image_status = annotations[image_name]["status"]
        
    #     if image_status == "completed_for_training" or image_status == "completed_for_testing":

    #         image_metrics[image_name] = {}

    #         pred_abs_boxes = np.array(image_predictions[image_name]["pred_image_abs_boxes"])
    #         pred_classes = np.array(image_predictions[image_name]["pred_classes"])
    #         pred_scores = np.array(image_predictions[image_name]["pred_scores"])

    #         pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
    #                                                                image_abs_boxes, image_classes)

    #         image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
    #         image_metric_fn.add(pred_for_mAP, true_for_mAP)
    #         # pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
    #         coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


    #         # image_metrics[image_name]["Image PASCAL VOC mAP"] = float(pascal_voc_mAP) * 100
    #         image_metrics[image_name]["MS COCO mAP"] = float(coco_mAP) * 100



    # return image_metrics


def get_AP_vals(annotated_boxes, predicted_boxes, predicted_scores):

    NUM_BOXES_THRESH = 100000

    if (annotated_boxes.shape[0] * predicted_boxes.shape[0]) < (NUM_BOXES_THRESH * NUM_BOXES_THRESH):

        annotated_classes = np.zeros(shape=(annotated_boxes.shape[0]))
        predicted_classes = np.zeros(shape=(predicted_boxes.shape[0]))

        pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
            predicted_boxes, 
            predicted_classes, 
            predicted_scores,
            annotated_boxes,
            annotated_classes)

        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        metric_fn.add(pred_for_mAP, true_for_mAP)

        ms_coco_mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        mAP_IoU_50 = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        mAP_IoU_75 = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        return {
            "AP (IoU=.50:.05:.95)": float(ms_coco_mAP) * 100,
            "AP (IoU=.50)": float(mAP_IoU_50) * 100,
            "AP (IoU=.75)": float(mAP_IoU_75) * 100
        }
    
    else:

        min_y_annotated = np.min(annotated_boxes[:, 0])
        min_x_annotated = np.min(annotated_boxes[:, 1])
        max_y_annotated = np.max(annotated_boxes[:, 2])
        max_x_annotated = np.max(annotated_boxes[:, 3])

        min_y_predicted = np.min(predicted_boxes[:, 0])
        min_x_predicted = np.min(predicted_boxes[:, 1])
        max_y_predicted = np.max(predicted_boxes[:, 2])
        max_x_predicted = np.max(predicted_boxes[:, 3])

        min_y = min(min_y_annotated, min_y_predicted)
        min_x = min(min_x_annotated, min_x_predicted)
        max_y = max(max_y_annotated, max_y_predicted)
        max_x = max(max_x_annotated, max_x_predicted)

        estimate_change = np.inf
        delta = 0.01
        ms_coco_mAP_vals = []
        mAP_IoU_50_vals = []
        mAP_IoU_75_vals = []
        while len(ms_coco_mAP_vals) < 10: # estimate_change > delta:
            point_y = random.randrange(min_y, max_y)
            point_x = random.randrange(min_x, max_x)

            sample_min_y = point_y - 1
            sample_min_x = point_x - 1
            sample_max_y = point_y + 1
            sample_max_x = point_x + 1

            prev_sample_region = [sample_min_y, sample_min_x, sample_max_y, sample_max_x]
            sample_region = prev_sample_region
            contained_annotated = get_contained_inds(annotated_boxes, [sample_region])
            contained_predicted = get_contained_inds(predicted_boxes, [sample_region])
            num_sample_boxes = contained_annotated * contained_predicted
            while (num_sample_boxes < (NUM_BOXES_THRESH * NUM_BOXES_THRESH)):
                cur_h = sample_max_y - sample_min_y
                sample_min_y = round(sample_min_y - (cur_h / 2))
                sample_max_y = round(sample_min_y + (cur_h / 2))

                cur_w = sample_max_x - sample_min_x
                sample_min_x = round(sample_min_x - (cur_w / 2))
                sample_max_x = round(sample_min_x + (cur_w / 2))

                prev_sample_region = sample_region
                sample_region = [sample_min_y, sample_min_x, sample_max_y, sample_max_x]
                contained_annotated = get_contained_inds(annotated_boxes, [sample_region])
                contained_predicted = get_contained_inds(predicted_boxes, [sample_region])
                num_sample_boxes = contained_annotated * contained_predicted

            contained_annotated = get_contained_inds(annotated_boxes, [prev_sample_region])
            contained_predicted = get_contained_inds(predicted_boxes, [prev_sample_region])

            print("Calculating sample mAP with {} annotated boxes and {} predicted boxes".format(contained_annotated.size, contained_predicted.size))

            sample_annotated_boxes = annotated_boxes[contained_annotated]
            sample_predicted_boxes = predicted_boxes[contained_predicted]
            sample_predicted_scores = predicted_scores[contained_predicted]

            sample_annotated_classes = np.zeros(shape=(sample_annotated_boxes.shape[0]))
            sample_predicted_classes = np.zeros(shape=(sample_predicted_boxes.shape[0]))

            pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
                sample_predicted_boxes, 
                sample_predicted_classes, 
                sample_predicted_scores,
                sample_annotated_boxes,
                sample_annotated_classes)
            metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
            metric_fn.add(pred_for_mAP, true_for_mAP)

            ms_coco_mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
            mAP_IoU_50 = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
            mAP_IoU_75 = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


            ms_coco_mAP_vals.append(ms_coco_mAP)
            mAP_IoU_50_vals.append(mAP_IoU_50)
            mAP_IoU_75_vals.append(mAP_IoU_75)

            print("ms_coco_mAP_vals", ms_coco_mAP_vals)
            print("mAP_IoU_50_vals", mAP_IoU_50_vals)
            print("mAP_IoU_75_vals", mAP_IoU_75_vals)



        return {
            "AP (IoU=.50:.05:.95)": float(np.mean(ms_coco_mAP_vals)) * 100,
            "AP (IoU=.50)": float(np.mean(mAP_IoU_50_vals)) * 100,
            "AP (IoU=.75)": float(np.mean(mAP_IoU_75_vals)) * 100
        }

def can_calculate_density(metadata, camera_specs):

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]

    if metadata["is_ortho"] == "yes":
        return False

    if (metadata["missing"]["latitude"] or metadata["missing"]["longitude"]) or metadata["camera_height"] == "":
        return False

    if make not in camera_specs:
        return False
    
    if model not in camera_specs[make]:
        return False

    return True

def calculate_area_m2(camera_specs, metadata, image_name):

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]
    camera_entry = camera_specs[make][model]

    gsd_h = (metadata["camera_height"] * camera_entry["sensor_height"]) / \
            (camera_entry["focal_length"] * metadata["images"][image_name]["height_px"])

    gsd_w = (metadata["camera_height"] * camera_entry["sensor_width"]) / \
            (camera_entry["focal_length"] * metadata["images"][image_name]["width_px"])

    gsd = min(gsd_h, gsd_w)

    area_m2 = (metadata["images"][image_name]["height_px"] * gsd) * (metadata["images"][image_name]["width_px"] * gsd)

    return area_m2
    


def create_spreadsheet(username, farm_name, field_name, mission_date, results_timestamp, download_uuid, annotation_version):



    image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                 farm_name, field_name, mission_date)

    results_dir = os.path.join(image_set_dir, "model", "results", results_timestamp)

    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)
    






    predictions_path = os.path.join(results_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path) #w3c_io.load_predictions(predictions_path, {"plant": 0})

    full_predictions_path = os.path.join(results_dir, "full_predictions.json")
    full_predictions = annotation_utils.load_predictions(full_predictions_path)


    if annotation_version == "preserved":
        annotations_path = os.path.join(results_dir, "annotations.json")
        # excess_green_record_path = os.path.join(results_dir, "excess_green_record.json")
        vegetation_record_path = os.path.join(results_dir, "vegetation_record.json")
        metrics_path = os.path.join(results_dir, "metrics.json")
        metrics = json_io.load_json(metrics_path)
    else:
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        # excess_green_record_path = os.path.join(image_set_dir, "excess_green", "record.json")
        vegetation_record_path = os.path.join(image_set_dir, "excess_green", "vegetation_record.json")


    annotations = annotation_utils.load_annotations(annotations_path) #w3c_io.load_annotations(annotations_path, {"plant": 0})
    # excess_green_record = json_io.load_json(excess_green_record_path)
    vegetation_record = json_io.load_json(vegetation_record_path)
    # if os.path.exists(excess_green_record_path):
    #     excess_green_record = json_io.load_json(excess_green_record_path)
    # else:
    #     excess_green_record = None


    args = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
        "predictions": predictions,
        "full_predictions": full_predictions,
        "annotations": annotations,
        "metadata": metadata,
        "camera_specs": camera_specs,
        # "excess_green_record": excess_green_record,
        "vegetation_record": vegetation_record
    }
    if annotation_version == "preserved":
        updated_metrics = metrics
    else:
        updated_metrics = collect_image_set_metrics(full_predictions, annotations)

    images_df = create_images_sheet(args, updated_metrics)
    regions_df = create_regions_sheet(args, updated_metrics)

    pandas.io.formats.excel.ExcelFormatter.header_style = None

    out_dir = os.path.join(results_dir, "retrieval", download_uuid)
    os.makedirs(out_dir)

    out_path = os.path.join(out_dir, "results.xlsx")
    # with pd.ExcelWriter(out_path) as writer:
    #     images_df.to_excel(writer, sheet_name="Images")


    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    fmt = writer.book.add_format({"font_name": "Courier New"})

    images_df.to_excel(writer, index=False, sheet_name="Images", na_rep='NA')  # send df to writer
    worksheet = writer.sheets["Images"]  # pull worksheet object

    worksheet.set_column('A:Z', None, fmt)
    worksheet.set_row(0, None, fmt)

    for idx, col in enumerate(images_df):  # loop through all columns
        series = images_df[col]
        if series.size > 0:
            max_entry_size = series.astype(str).map(len).max()
        else:
            max_entry_size = 0
        max_len = max((
            max_entry_size,  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width


    regions_df.to_excel(writer, index=False, sheet_name="Regions", na_rep='NA')  # send df to writer
    worksheet = writer.sheets["Regions"]  # pull worksheet object

    worksheet.set_column('A:Z', None, fmt)
    worksheet.set_row(0, None, fmt)

    for idx, col in enumerate(regions_df):  # loop through all columns
        series = regions_df[col]
        if series.size > 0:
            max_entry_size = series.astype(str).map(len).max()
        else:
            max_entry_size = 0
        max_len = max((
            max_entry_size,  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width



    writer.save()



    # df = pd.DataFrame(data=d, columns=columns)
    # df.sort_values(by="image_name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_name"])))

    # out_dir = os.path.join(results_dir, "retrieval", download_uuid)
    # os.makedirs(out_dir)

    # out_path = os.path.join(out_dir, "results.csv")
    # df.to_csv(out_path, index=False)
    
    metrics_out_path = os.path.join(out_dir, "metrics.json")
    json_io.save_json(metrics_out_path, updated_metrics)





def create_images_sheet(args, updated_metrics):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    predictions = args["predictions"]
    # full_predictions = args["full_predictions"]
    annotations = args["annotations"]
    metadata = args["metadata"]
    camera_specs = args["camera_specs"]
    # excess_green_record = args["excess_green_record"]
    vegetation_record = args["vegetation_record"]


    include_density = can_calculate_density(metadata, camera_specs)


    # defines the order of the columns
    columns = [
        "username",
        "farm_name",
        "field_name",
        "mission_date",
        "image_name",
        "training_regions",
        "test_regions",
        "image_is_fully_annotated",
        "source_of_annotations",
        "annotated_plant_count",
        "predicted_plant_count"
    ]

    if include_density:
        columns.extend(["annotated_plant_count_per_square_metre", "predicted_plant_count_per_square_metre"])

    # if excess_green_record is not None:
    columns.extend([
        "excess_green_threshold",
        "vegetation_percentage"
    ])

    # columns.append("MS_COCO_mAP")

    metrics_lst = [
        "AP (IoU=.50:.05:.95)",
        "AP (IoU=.50)",
        "AP (IoU=.75)",
        "F1 Score (IoU=.50, conf>=.50)",
        "F1 Score (IoU=.75, conf>=.50)"
    ]
    columns.extend(metrics_lst)

    d = {}
    for c in columns:
        d[c] = []


    # new_metrics = {}

    for image_name in annotations.keys():

        image_abs_boxes = annotations[image_name]["boxes"]
        training_regions = annotations[image_name]["training_regions"]
        test_regions = annotations[image_name]["test_regions"]

        # if annotations[image_name]["predictions_used_as_annotations"]:
        #     predictions_used_as_annotations = "yes"
        # else:
        #     predictions_used_as_annotations = "no"
        annotations_source = annotations[image_name]["source"]


        image_height_px = metadata["images"][image_name]["height_px"]
        image_width_px = metadata["images"][image_name]["width_px"]

        if annotation_utils.is_fully_annotated_for_training(annotations, image_name, image_width_px, image_height_px):
            fully_annotated = "yes: for fine-tuning"

        elif annotation_utils.is_fully_annotated_for_testing(annotations, image_name, image_width_px, image_height_px):
            fully_annotated = "yes: for testing"
            
        else:
            fully_annotated = "no"
        # image_status = annotations[image_name]["status"]

        # pred_image_abs_boxes = predictions[image_name]["boxes"]
        pred_image_scores = predictions[image_name]["scores"]
        # print(pred_image_scores)

        annotated_count = image_abs_boxes.shape[0]
        predicted_count = np.sum(pred_image_scores >= 0.50)

        d["username"].append(username)
        d["farm_name"].append(farm_name)
        d["field_name"].append(field_name)
        d["mission_date"].append(mission_date)
        d["image_name"].append(image_name)
        d["training_regions"].append(len(training_regions))
        d["test_regions"].append(len(test_regions))
        d["image_is_fully_annotated"].append(fully_annotated)
        d["source_of_annotations"].append(annotations_source)
        # if image_status == "unannotated":
        #     d["annotated_plant_count"].append("NA")
        # else:
        #     d["annotated_plant_count"].append(annotated_count)
        d["annotated_plant_count"].append(annotated_count)


        d["predicted_plant_count"].append(predicted_count)


        if include_density:
            area_m2 = calculate_area_m2(camera_specs, metadata, image_name)
            # if image_status == "unannotated":
            #     d["annotated_plant_count_per_square_metre"].append("NA")
            # else:
            d["annotated_plant_count_per_square_metre"].append(round(annotated_count / area_m2, 2))
            d["predicted_plant_count_per_square_metre"].append(round(predicted_count / area_m2, 2))


        # if excess_green_record is not None:
        d["excess_green_threshold"].append(vegetation_record[image_name]["sel_val"])
        d["vegetation_percentage"].append(vegetation_record[image_name]["image"])

        if fully_annotated == "no":
            for metric in metrics_lst:
                d[metric].append("NA")
        elif fully_annotated == "yes: for fine-tuning":
            for metric in metrics_lst:
                d[metric].append(round(updated_metrics[metric][image_name]["training_regions"][0], 2))
        else:
            for metric in metrics_lst:
                d[metric].append(round(updated_metrics[metric][image_name]["test_regions"][0], 2))


        # if annotation_version == "preserved":
        #     if image_name in metrics:
        #         ms_coco_mAP = round(metrics[image_name]["MS COCO mAP"], 2)

        #         new_metrics[image_name] = {}
        #         new_metrics[image_name]["MS COCO mAP"] = round(metrics[image_name]["MS COCO mAP"], 2)
        #     else:
        #         ms_coco_mAP = "NA"

        # else:
        #     if image_status == "completed_for_training" or image_status == "completed_for_testing":
        #         ms_coco_mAP = round(calculate_MS_COCO_mAP_for_image(predictions, annotations, image_name), 2)


        #         new_metrics[image_name] = {}
        #         new_metrics[image_name]["MS COCO mAP"] = round(calculate_MS_COCO_mAP_for_image(predictions, annotations, image_name), 2)
        #     else:
        #         ms_coco_mAP = "NA"


        # d["MS_COCO_mAP"].append(ms_coco_mAP)
    print(d)
    df = pd.DataFrame(data=d, columns=columns)
    df.sort_values(by="image_name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_name"])))
    return df


def create_regions_sheet(args, updated_metrics):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    predictions = args["predictions"]
    annotations = args["annotations"]
    metadata = args["metadata"]
    camera_specs = args["camera_specs"]
    vegetation_record = args["vegetation_record"]


    include_density = can_calculate_density(metadata, camera_specs)


    # defines the order of the columns
    columns = [
        "username",
        "farm_name",
        "field_name",
        "mission_date",
        "image_name",
        "region_name",
        "source_of_annotations_for_image",
        "annotated_plant_count",
        "predicted_plant_count"
    ]

    if include_density:
        columns.extend(["annotated_plant_count_per_square_metre", "predicted_plant_count_per_square_metre"])

    # if excess_green_record is not None:
    # columns.append("ground_cover_percentage")
    columns.extend([
        "excess_green_threshold",
        "vegetation_percentage"
    ])

    metrics_lst = [
        "AP (IoU=.50:.05:.95)",
        "AP (IoU=.50)",
        "AP (IoU=.75)",
        "F1 Score (IoU=.50, conf>=.50)",
        "F1 Score (IoU=.75, conf>=.50)"
    ]
    columns.extend(metrics_lst)

    d = {}
    for c in columns:
        d[c] = []



    for image_name in annotations.keys():

        annotated_boxes = annotations[image_name]["boxes"]
        predicted_boxes = predictions[image_name]["boxes"]
        predicted_scores = predictions[image_name]["scores"]

        annotations_source = annotations[image_name]["source"]
        # above_thresh_predicted_boxes = predicted_boxes[predicted_scores >= 0.50]

        # metrics["MS COCO mAP"][image_name] = {}
        # metrics["F1 Score (IoU=0.5)"][image_name] = {}
        # metrics["F1 Score (IoU=0.7)"][image_name] = {}
        # metrics["F1 Score (IoU=0.9)"][image_name] = {}

        for region_type in ["training", "test"]:

            # metrics["MS COCO mAP"][image_name][region_type + "_regions"] = []
            # metrics["F1 Score (IoU=0.5)"][image_name][region_type + "_regions"] = []
            # metrics["F1 Score (IoU=0.7)"][image_name][region_type + "_regions"] = []
            # metrics["F1 Score (IoU=0.9)"][image_name][region_type + "_regions"] = []

            regions = annotations[image_name][region_type + "_regions"]

            for i, region in enumerate(regions):

                if region_type == "training":
                    region_name = "fine_tuning_" + (str(i+1))
                else:
                    region_name = "test_" + (str(i+1))

                annotated_inds = box_utils.get_contained_inds(annotated_boxes, [region])
                region_annotated_boxes = annotated_boxes[annotated_inds]
                predicted_inds = box_utils.get_contained_inds(predicted_boxes, [region])
                # region_predicted_boxes = predicted_boxes[predicted_inds]
                region_predicted_scores = predicted_scores[predicted_inds]


                annotated_count = region_annotated_boxes.shape[0]
                predicted_count = np.sum(region_predicted_scores >= 0.50)

                d["username"].append(username)
                d["farm_name"].append(farm_name)
                d["field_name"].append(field_name)
                d["mission_date"].append(mission_date)
                d["image_name"].append(image_name)
                d["region_name"].append(region_name)
                d["source_of_annotations_for_image"].append(annotations_source)
                d["annotated_plant_count"].append(annotated_count)
                d["predicted_plant_count"].append(predicted_count)


                if include_density:
                    area_m2 = calculate_area_m2(camera_specs, metadata, image_name)
                    d["annotated_plant_count_per_square_metre"].append(round(annotated_count / area_m2, 2))
                    d["predicted_plant_count_per_square_metre"].append(round(predicted_count / area_m2, 2))



                d["excess_green_threshold"].append(vegetation_record[image_name]["sel_val"])
                d["vegetation_percentage"].append(vegetation_record[image_name][region_type + "_regions"][i])
                # MS_COCO_mAP = get_MS_COCO_mAP(region_annotated_boxes, region_predicted_boxes, region_predicted_scores)
                # d["MS_COCO_mAP"].append(round(MS_COCO_mAP, 2))

                for metric in metrics_lst:
                    d[metric].append(round(updated_metrics[metric][image_name][region_type + "_regions"][i], 2))

                # metrics["MS COCO mAP"][image_name][region_type + "_regions"].append(MS_COCO_mAP)

                # sel_region_predicted_boxes = region_predicted_boxes[region_predicted_scores >= 0.50]
                # f1_iou_05 = get_f1_score(region_annotated_boxes, sel_region_predicted_boxes, iou_thresh=0.5)
                # f1_iou_07 = get_f1_score(region_annotated_boxes, sel_region_predicted_boxes, iou_thresh=0.7)
                # f1_iou_09 = get_f1_score(region_annotated_boxes, sel_region_predicted_boxes, iou_thresh=0.9)
                # metrics["F1 Score (IoU=0.5)"][image_name][region_type + "_regions"].append(f1_iou_05)
                # metrics["F1 Score (IoU=0.7)"][image_name][region_type + "_regions"].append(f1_iou_07)
                # metrics["F1 Score (IoU=0.9)"][image_name][region_type + "_regions"].append(f1_iou_09)



    df = pd.DataFrame(data=d, columns=columns)
    df.sort_values(by="image_name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_name"])))
    return df #, metrics

# def prepare_report(out_path, farm_name, field_name, mission_date, 
#                    image_predictions, annotations, excess_green_record, metrics):

#     logger = logging.getLogger(__name__)
#     logger.info("Preparing report")

#     #num_classes = len(config["arch"]["class_map"].keys())



#     d = {
#         "farm_name": [],
#         "field_name": [],
#         "mission_date": [],
#         "image_status": [],
#         "image_name": [],
#         "annotated_plant_count": [],
#         "predicted_plant_count": [],
#         "ground_cover_percentage": [],
#         "MS_COCO_mAP": [],
#         #"PASCAL_VOC_mAP": []
#     }

#     # for class_name in config["arch"]["class_map"].keys():
#     #     d["annotated_" + class_name + "_count"] = []
#     #     d["model_" + class_name + "_count"] = []


#     for image_name in tqdm.tqdm(image_predictions.keys(), desc="Collecting metrics"):

#         image_abs_boxes = annotations[image_name]["boxes"]
#         image_status = annotations[image_name]["status"]


#         d["farm_name"].append(farm_name)
#         d["field_name"].append(field_name)
#         d["mission_date"].append(mission_date)
#         d["image_status"].append(image_status)
#         d["image_name"].append(image_name)

#         if image_status == "unannotated":
#             annotated_count = "NA"
#         else:
#             annotated_count = image_abs_boxes.shape[0]
#         pred_abs_boxes = np.array(image_predictions[image_name]["pred_image_abs_boxes"])
#         #pred_classes = np.array(image_predictions[image_name]["pred_classes"])
#         pred_scores = np.array(image_predictions[image_name]["pred_scores"])

#         mask = pred_scores >= 0.5

#         sel_pred_abs_boxes = pred_abs_boxes[mask]
#         #sel_pred_classes = pred_classes[mask]
#         sel_pred_scores = pred_scores[mask]

#         d["annotated_" + "plant" + "_count"].append(str(annotated_count))
#         d["predicted_" + "plant" + "_count"].append(str(sel_pred_abs_boxes.shape[0]))

#         d["ground_cover_percentage"].append("%.2f" % (excess_green_record[image_name]["ground_cover_percentage"]))

#         if image_name in metrics:
#             d["MS_COCO_mAP"].append(metrics[image_name]["Image MS COCO mAP"])
#         else:
#             d["MS_COCO_mAP"].append("NA")


#     pandas.io.formats.excel.ExcelFormatter.header_style = None
#     df = pd.DataFrame(data=d)
#     df.sort_values(by="image_name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_name"])))
#     df.to_csv(out_path, index=False)
    
#     # writer = pd.ExcelWriter(out_path, engine="xlsxwriter")

#     # df.to_excel(writer, index=False, sheet_name="Sheet1", na_rep='NA')  # send df to writer
#     # worksheet = writer.sheets["Sheet1"]  # pull worksheet object


#     # # for i, width in enumerate(get_col_widths(df)):
#     # #     worksheet.set_column(i, i, width)
#     # # for column in df:
#     # #     worksheet.column_dimensions[column].bestFit = True
#     # for idx, col in enumerate(df):  # loop through all columns
#     #     series = df[col]
#     #     max_len = max((
#     #         series.astype(str).map(len).max(),  # len of largest item
#     #         len(str(series.name))  # len of column name/header
#     #         )) # + 1  # adding a little extra space
#     #     worksheet.set_column(idx, idx, 0.9 * max_len)  # set column width
#     # writer.save()




# def collect_metrics(image_names, metrics, predictions, dataset, config,
#                     collect_patch_metrics=True, calculate_mAP=True):

#     logger = logging.getLogger(__name__)

#     num_classes = len(config["arch"]["class_map"].keys())

#     annotated_image_counts = {}
#     pred_image_counts = {}

#     class_map = config["arch"]["class_map"]
#     reverse_class_map =  config["arch"]["reverse_class_map"]

#     all_images_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes) #config.arch["num_classes"])

#     annotations = w3c_io.load_annotations(dataset.annotations_path, config["arch"]["class_map"])


#     # completed_images = w3c_io.get_completed_images(annotations)
#     # optimal_thresh_val, optimal_mean_abs_diff = calculate_optimal_score_threshold(annotations, predictions, completed_images)


    
#     point_metrics = metrics["point"]
#     boxplot_metrics = metrics["boxplot"]    
#     image_metrics = metrics["image"]
    
#     # point_metrics["true_optimal_score_threshold"] = {}
#     # point_metrics["true_optimal_score_threshold"]["threshold_value"] = optimal_thresh_val
#     # point_metrics["true_optimal_score_threshold"]["mean_absolute_difference"] = optimal_mean_abs_diff





#     annotated_image_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
#     pred_image_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}

#     #for image in dataset.completed_images:
#     for image_name in image_names: #predictions["image_predictions"].keys():

#         if annotations[image_name]["status"] == "completed":

#             #img_abs_boxes, img_classes = xml_io.load_boxes_and_classes(img.xml_path, img_set.class_map) #config.arch["class_map"])
#             image_abs_boxes = annotations[image_name]["boxes"]
#             image_classes = annotations[image_name]["classes"]
            
            
#             unique, counts = np.unique(image_classes, return_counts=True)
#             class_num_to_count = dict(zip(unique, counts))
#             cur_image_class_counts = {k: 0 for k in class_map} #config.arch["class_map"].keys()}
#             for class_num in class_num_to_count.keys():
#                 cur_image_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
#                 #cur_img_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

#             cur_image_pred_class_counts = predictions["image_predictions"][image_name]["pred_class_counts"]

#             for class_name in class_map.keys(): #config.arch["class_map"].keys():
#                 annotated_image_counts[class_name].append(cur_image_class_counts[class_name])
#                 pred_image_counts[class_name].append(cur_image_pred_class_counts[class_name])

#             #annotated_img_count = np.shape(img_abs_boxes)[0]

#             #pred_img_count = predictions["image_predictions"][img.img_name]["pred_count"]
            
#             if calculate_mAP:
#                 pred_abs_boxes = np.array(predictions["image_predictions"][image_name]["pred_image_abs_boxes"])
#                 pred_classes = np.array(predictions["image_predictions"][image_name]["pred_classes"])
#                 pred_scores = np.array(predictions["image_predictions"][image_name]["pred_scores"])

#                 pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
#                                                                     image_abs_boxes, image_classes)
#                 all_images_metric_fn.add(pred_for_mAP, true_for_mAP)


#                 image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
#                 image_metric_fn.add(pred_for_mAP, true_for_mAP)
#                 pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
#                 coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


#                 image_metrics[image_name]["Image PASCAL VOC mAP"] = float(pascal_voc_mAP) * 100
#                 image_metrics[image_name]["Image MS COCO mAP"] = float(coco_mAP) * 100

                
#                 #annotated_img_counts.append(annotated_img_count)
#             #pred_img_counts.append(pred_img_count)


#     #annotated_img_counts = np.array(annotated_img_counts)
#     #pred_img_counts = np.array(pred_img_counts)

#     # annotated_patch_counts = []
#     # pred_patch_counts = []

#     if collect_patch_metrics:
#         annotated_patch_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
#         pred_patch_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
#         for patch_name, patch_pred in predictions["patch_predictions"].items():

#             if "patch_classes" in patch_pred:
#                 patch_classes = patch_pred["patch_classes"]
#                 unique, counts = np.unique(patch_classes, return_counts=True)
#                 class_num_to_count = dict(zip(unique, counts))
#                 cur_patch_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
#                 for class_num in class_num_to_count.keys():
#                     cur_patch_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
#                     #cur_patch_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

#                 pred_patch_classes = patch_pred["pred_classes"]
#                 pred_unique, pred_counts = np.unique(pred_patch_classes, return_counts=True)
#                 class_num_to_pred_count = dict(zip(pred_unique, pred_counts))
#                 cur_patch_pred_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
#                 for class_num in class_num_to_pred_count.keys():
#                     cur_patch_pred_class_counts[reverse_class_map[class_num]] = class_num_to_pred_count[class_num]
#                     #cur_patch_pred_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_pred_count[class_num]

#                 for class_name in class_map.keys(): #config.arch["class_map"].keys():
#                     annotated_patch_counts[class_name].append(cur_patch_class_counts[class_name])
#                     pred_patch_counts[class_name].append(cur_patch_pred_class_counts[class_name])


#     print("annotated_image_counts", annotated_image_counts)
#     total_obj_count = float(np.sum([np.sum(np.array(annotated_image_counts[class_name])) for class_name in annotated_image_counts.keys()]))

#     logger.info("Started calculating count metrics.")

#     # if "metrics" not in predictions:
#     #     predictions["metrics"] = {}

#     # if "point" not in predictions["metrics"]:
#     #     predictions["metrics"]["point"] = {}

#     # if "boxplot" not in predictions["metrics"]:
#     #     predictions["metrics"]["boxplot"] = {}




#     point_metrics["Image Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#     point_metrics["Image Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#     point_metrics["Image R Squared"] = {"Cross-Class Weighted Average": 0}
#     point_metrics["Image Non-Zero Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#     if collect_patch_metrics:
#         point_metrics["Patch Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#         point_metrics["Patch Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#         point_metrics["Patch R Squared"] = {"Cross-Class Weighted Average": 0}
#         point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}


#     boxplot_metrics["Difference in Count (Image)"] = {}
#     boxplot_metrics["Absolute Difference in Count (Image)"] = {} #{"Cross-Class Weighted Average": 0}
#     boxplot_metrics["Percent Difference in Count (Image)"] = {} #{"Cross-Class Weighted Average": 0}
#     #boxplot_metrics["Diff. in Count (Patch)"] = {}
#     #boxplot_metrics["Absolute Diff. in Count (Patch)"] = {} #{"Cross-Class Weighted Average": 0}
#     #boxplot_metrics["Percent Diff. in Count (Patch)"] = {} #{"Cross-Class Weighted Average": 0}
    
#     # boxplot_metrics["Confidence"] = {"Cross-Class Weighted Average": 0}
#     # boxplot_metrics["Box Area"] = {"Cross-Class Weighted Average": 0}
#     # boxplot_metrics["Inference Time (Per Image)"] = {"Cross-Class Weighted Average": 0}
#     # boxplot_metrics["Inference Time (Per Patch)"] = {"Cross-Class Weighted Average": 0}
    
#     #predictions["metrics"]["Patch Five Num. Summary Diff. in Count"] = {}
#     #predictions["metrics"]["Diff. in Count Occurrences"] = {}

#     for class_name in annotated_image_counts.keys():

#         image_class_annotated_count = np.array(annotated_image_counts[class_name])
#         image_class_pred_count = np.array(pred_image_counts[class_name])

#         point_metrics["Image Mean Abs. Diff. in Count"][class_name] = \
#                         float(np.mean(abs_DiC(image_class_annotated_count, image_class_pred_count)))
#         point_metrics["Image Mean Sq. Diff. in Count"][class_name] = \
#                         float(np.mean(squared_DiC(image_class_annotated_count, image_class_pred_count)))
#         point_metrics["Image R Squared"][class_name] = \
#                         r_squared(image_class_annotated_count, image_class_pred_count)
#         point_metrics["Image Non-Zero Mean Abs. Diff. in Count"][class_name] = \
#                         float(np.mean(nonzero_abs_DiC(image_class_annotated_count, image_class_pred_count)))


        



#         #predictions["metrics"]["Patch Five Num. Summary Diff. in Count"][class_name] = five_num_summary_DiC(patch_class_annotated_count, patch_class_pred_count)
#         #predictions["metrics"]["Diff. in Count Occurrences"][class_name] = DiC_occurrences(patch_class_annotated_count, patch_class_pred_count)
        
#         boxplot_metrics["Difference in Count (Image)"][class_name] = \
#                     boxplot_data(DiC(image_class_annotated_count, image_class_pred_count))

#         boxplot_metrics["Absolute Difference in Count (Image)"][class_name] = \
#                     boxplot_data(abs_DiC(image_class_annotated_count, image_class_pred_count))

#         boxplot_metrics["Percent Difference in Count (Image)"][class_name] = \
#                     boxplot_data(pct_DiC(image_class_annotated_count, image_class_pred_count))

#         # boxplot_metrics["Diff. in Count (Patch)"][class_name] = \
#         #             boxplot_data(DiC(patch_class_annotated_count, patch_class_pred_count))

#         # boxplot_metrics["Absolute Diff. in Count (Patch)"][class_name] = \
#         #             boxplot_data(abs_DiC(patch_class_annotated_count, patch_class_pred_count))

#         # boxplot_metrics["Percent Diff. in Count (Patch)"][class_name] = \
#         #             boxplot_data(pct_DiC(patch_class_annotated_count, patch_class_pred_count))



        
#         total_class_annotated_count = float(np.sum(image_class_annotated_count))

#         point_metrics["Image Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image Mean Abs. Diff. in Count"][class_name]
#         point_metrics["Image Mean Sq. Diff. in Count"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image Mean Sq. Diff. in Count"][class_name]
#         point_metrics["Image R Squared"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image R Squared"][class_name]
#         point_metrics["Image Non-Zero Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image Non-Zero Mean Abs. Diff. in Count"][class_name]



#         if collect_patch_metrics:

#             patch_class_annotated_count = np.array(annotated_patch_counts[class_name])
#             patch_class_pred_count = np.array(pred_patch_counts[class_name])

#             point_metrics["Patch Mean Abs. Diff. in Count"][class_name] = \
#                             float(np.mean(abs_DiC(patch_class_annotated_count, patch_class_pred_count)))
#             point_metrics["Patch Mean Sq. Diff. in Count"][class_name] = \
#                             float(np.mean(squared_DiC(patch_class_annotated_count, patch_class_pred_count)))
#             point_metrics["Patch R Squared"][class_name] = \
#                             r_squared(patch_class_annotated_count, patch_class_pred_count)
#             point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"][class_name] = \
#                             float(np.mean(nonzero_abs_DiC(patch_class_annotated_count, patch_class_pred_count)))

#             point_metrics["Patch Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch Mean Abs. Diff. in Count"][class_name]
#             point_metrics["Patch Mean Sq. Diff. in Count"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch Mean Sq. Diff. in Count"][class_name]
#             point_metrics["Patch R Squared"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch R Squared"][class_name]
#             point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"][class_name]




#     logger.info("Finished calculating count metrics.")

#     if calculate_mAP:
#         logger.info("Started calculating mAP scores.")

#         pascal_voc_mAP = all_images_metric_fn.value(iou_thresholds=0.5)['mAP']
#         coco_mAP = all_images_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

#         logger.info("Finished calculating mAP scores.")

#         point_metrics["Image PASCAL VOC mAP"] = {}
#         point_metrics["Image PASCAL VOC mAP"]["---"] = float(pascal_voc_mAP) * 100
#         point_metrics["Image MS COCO mAP"] = {}
#         point_metrics["Image MS COCO mAP"]["---"] = float(coco_mAP) * 100




def get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                              true_abs_boxes, true_classes):

    if pred_abs_boxes.size > 0:
        pred_abs_boxes = box_utils.swap_xy_np(pred_abs_boxes)
    else:
        pred_abs_boxes = np.reshape(pred_abs_boxes, (0, 4)) #np.expand_dims(pred_abs_boxes, axis=-1)
        
    pred_classes = np.expand_dims(pred_classes, axis=-1)
    pred_scores = np.expand_dims(pred_scores, axis=-1)
    pred = np.hstack([pred_abs_boxes, pred_classes, pred_scores])

    if true_abs_boxes.size > 0:
        true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    else:
        #true_abs_boxes = np.expand_dims(true_abs_boxes, axis=-1)
        true_abs_boxes = np.reshape(true_abs_boxes, (0, 4)) 
    #true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    true_classes = np.expand_dims(true_classes, axis=-1)
    difficult = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    crowd = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    true = np.hstack([true_abs_boxes, true_classes, difficult, crowd])  

    return pred, true





