import logging
import tqdm
import os
import numpy as np
from mean_average_precision import MetricBuilder


import models.common.box_utils as box_utils

from io_utils import json_io, xml_io


def mean_DiC(actual, pred):
    return np.mean(actual - pred)

def mean_abs_DiC(actual, pred):
    return float(np.mean(np.abs(actual - pred)))

def mean_squared_DiC(actual, pred):
    return float(np.mean((actual - pred) ** 2))

def r_squared(actual, pred):
    SS_res = np.sum((actual - pred) ** 2)
    SS_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - (SS_res / SS_tot))

def five_num_summary(data):
    return np.percentile(np.array(data), [0, 25, 50, 75, 100], interpolation='midpoint').tolist()

def five_num_summary_DiC(actual, pred):
    return five_num_summary(actual - pred)

def DiC_occurrences(actual, pred):
    a = (actual - pred)
    vals = np.arange(a.min(), a.max() + 1)
    occurrences, _ = np.histogram(a, bins=(a.max() - a.min() + 1))
    return {
            "vals": vals.tolist(),
            "occurrences": occurrences.tolist()
           }

def collect_metrics(predictions, img_dataset, config):

    logger = logging.getLogger(__name__)

    annotated_img_counts = {}
    pred_img_counts = {}

    img_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=config.arch["num_classes"])


    annotated_img_counts = {k: [] for k in config.arch["class_map"].keys()}
    pred_img_counts = {k: [] for k in config.arch["class_map"].keys()}
    for img in img_dataset.imgs:

        img_abs_boxes, img_classes = xml_io.load_boxes_and_classes(img.xml_path, config.arch["class_map"])
        unique, counts = np.unique(img_classes, return_counts=True)
        class_num_to_count = dict(zip(unique, counts))
        cur_img_class_counts = {k: 0 for k in config.arch["class_map"].keys()}
        for class_num in class_num_to_count.keys():
            cur_img_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

        cur_img_pred_class_counts = predictions["image_predictions"][img.img_name]["pred_class_counts"]

        for class_name in config.arch["class_map"].keys():
            annotated_img_counts[class_name].append(cur_img_class_counts[class_name])
            pred_img_counts[class_name].append(cur_img_pred_class_counts[class_name])

        #annotated_img_count = np.shape(img_abs_boxes)[0]

        #pred_img_count = predictions["image_predictions"][img.img_name]["pred_count"]
        pred_abs_boxes = np.array(predictions["image_predictions"][img.img_name]["pred_img_abs_boxes"])
        pred_classes = np.array(predictions["image_predictions"][img.img_name]["pred_classes"])
        pred_scores = np.array(predictions["image_predictions"][img.img_name]["pred_scores"])

        pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                                                               img_abs_boxes, img_classes)
        img_metric_fn.add(pred_for_mAP, true_for_mAP)

        #annotated_img_counts.append(annotated_img_count)
        #pred_img_counts.append(pred_img_count)


    #annotated_img_counts = np.array(annotated_img_counts)
    #pred_img_counts = np.array(pred_img_counts)

    # annotated_patch_counts = []
    # pred_patch_counts = []

    annotated_patch_counts = {k: [] for k in config.arch["class_map"].keys()}
    pred_patch_counts = {k: [] for k in config.arch["class_map"].keys()}
    for patch_name, patch_pred in predictions["patch_predictions"].items():

        patch_classes = patch_pred["patch_classes"]
        unique, counts = np.unique(patch_classes, return_counts=True)
        class_num_to_count = dict(zip(unique, counts))
        cur_patch_class_counts = {k: 0 for k in config.arch["class_map"].keys()}
        for class_num in class_num_to_count.keys():
            cur_patch_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

        pred_patch_classes = patch_pred["pred_classes"]
        pred_unique, pred_counts = np.unique(pred_patch_classes, return_counts=True)
        class_num_to_pred_count = dict(zip(pred_unique, pred_counts))
        cur_patch_pred_class_counts = {k: 0 for k in config.arch["class_map"].keys()}
        for class_num in class_num_to_pred_count.keys():
            cur_patch_pred_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_pred_count[class_num]

        for class_name in config.arch["class_map"].keys():
            annotated_patch_counts[class_name].append(cur_patch_class_counts[class_name])
            pred_patch_counts[class_name].append(cur_patch_pred_class_counts[class_name])



    total_obj_count = float(np.sum([np.sum(np.array(annotated_img_counts[class_name])) for class_name in annotated_img_counts.keys()]))

    logger.info("Started calculating count metrics.")


    predictions["metrics"]["Image Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Sum": 0}
    predictions["metrics"]["Image Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Sum": 0}
    predictions["metrics"]["Image R Squared"] = {"Cross-Class Weighted Sum": 0}

    predictions["metrics"]["Patch Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Sum": 0}
    predictions["metrics"]["Patch Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Sum": 0}
    predictions["metrics"]["Patch R Squared"] = {"Cross-Class Weighted Sum": 0}
    predictions["metrics"]["Patch Five Num. Summary Diff. in Count"] = {}
    predictions["metrics"]["Diff. in Count Occurrences"] = {}

    for class_name in annotated_img_counts.keys():

        img_class_annotated_count = np.array(annotated_img_counts[class_name])
        img_class_pred_count = np.array(pred_img_counts[class_name])

        predictions["metrics"]["Image Mean Abs. Diff. in Count"][class_name] = mean_abs_DiC(img_class_annotated_count, img_class_pred_count)
        predictions["metrics"]["Image Mean Sq. Diff. in Count"][class_name] = mean_squared_DiC(img_class_annotated_count, img_class_pred_count)
        predictions["metrics"]["Image R Squared"][class_name] = r_squared(img_class_annotated_count, img_class_pred_count)


        patch_class_annotated_count = np.array(annotated_patch_counts[class_name])
        patch_class_pred_count = np.array(pred_patch_counts[class_name])

        predictions["metrics"]["Patch Mean Abs. Diff. in Count"][class_name] = mean_abs_DiC(patch_class_annotated_count, patch_class_pred_count)
        predictions["metrics"]["Patch Mean Sq. Diff. in Count"][class_name] = mean_squared_DiC(patch_class_annotated_count, patch_class_pred_count)
        predictions["metrics"]["Patch R Squared"][class_name] = r_squared(patch_class_annotated_count, patch_class_pred_count)

        predictions["metrics"]["Patch Five Num. Summary Diff. in Count"][class_name] = five_num_summary_DiC(patch_class_annotated_count, patch_class_pred_count)
        predictions["metrics"]["Diff. in Count Occurrences"][class_name] = DiC_occurrences(patch_class_annotated_count, patch_class_pred_count)


        
        total_class_annotated_count = float(np.sum(img_class_annotated_count))

        predictions["metrics"]["Image Mean Abs. Diff. in Count"]["Cross-Class Weighted Sum"] += \
            (total_class_annotated_count / total_obj_count) * predictions["metrics"]["Image Mean Abs. Diff. in Count"][class_name]
        predictions["metrics"]["Image Mean Sq. Diff. in Count"]["Cross-Class Weighted Sum"] += \
            (total_class_annotated_count / total_obj_count) * predictions["metrics"]["Image Mean Sq. Diff. in Count"][class_name]
        predictions["metrics"]["Image R Squared"]["Cross-Class Weighted Sum"] += \
            (total_class_annotated_count / total_obj_count) * predictions["metrics"]["Image R Squared"][class_name]


        predictions["metrics"]["Patch Mean Abs. Diff. in Count"]["Cross-Class Weighted Sum"] += \
            (total_class_annotated_count / total_obj_count) * predictions["metrics"]["Patch Mean Abs. Diff. in Count"][class_name]
        predictions["metrics"]["Patch Mean Sq. Diff. in Count"]["Cross-Class Weighted Sum"] += \
            (total_class_annotated_count / total_obj_count) * predictions["metrics"]["Patch Mean Sq. Diff. in Count"][class_name]
        predictions["metrics"]["Patch R Squared"]["Cross-Class Weighted Sum"] += \
            (total_class_annotated_count / total_obj_count) * predictions["metrics"]["Patch R Squared"][class_name]




    logger.info("Finished calculating count metrics.")


    logger.info("Started calculating mAP scores.")

    pascal_voc_mAP = img_metric_fn.value(iou_thresholds=0.5)['mAP']
    coco_mAP = img_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

    logger.info("Finished calculating mAP scores.")

    predictions["metrics"]["Image PASCAL VOC mAP"] = {}
    predictions["metrics"]["Image PASCAL VOC mAP"]["---"] = float(pascal_voc_mAP)
    predictions["metrics"]["Image MS COCO mAP"] = {}
    predictions["metrics"]["Image MS COCO mAP"]["---"] = float(coco_mAP)




def get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                              true_abs_boxes, true_classes):

    if pred_abs_boxes.size > 0:
        pred_abs_boxes = box_utils.swap_xy_np(pred_abs_boxes)
    else:
        pred_abs_boxes = np.expand_dims(pred_abs_boxes, axis=-1)
    pred_classes = np.expand_dims(pred_classes, axis=-1)
    pred_scores = np.expand_dims(pred_scores, axis=-1)
    pred = np.hstack([pred_abs_boxes, pred_classes, pred_scores])

    true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    true_classes = np.expand_dims(true_classes, axis=-1)
    difficult = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    crowd = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    true = np.hstack([true_abs_boxes, true_classes, difficult, crowd])  

    return pred, true