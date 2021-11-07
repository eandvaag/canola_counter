import logging
import tqdm
import os
import numpy as np
from mean_average_precision import MetricBuilder


import models.common.box_utils as box_utils
#import models.common.decode_predictions as decode_predictions

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

    img_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=config.num_classes)


    annotated_img_counts = {k: [] for k in config.class_map.keys()}
    pred_img_counts = {k: [] for k in config.class_map.keys()}
    for img in img_dataset.imgs:

        img_abs_boxes, img_classes = xml_io.load_boxes_and_classes(img.xml_path, config.class_map)
        unique, counts = np.unique(img_classes, return_counts=True)
        class_num_to_count = dict(zip(unique, counts))
        cur_img_class_counts = {k: 0 for k in config.class_map.keys()}
        for class_num in class_num_to_count.keys():
            cur_img_class_counts[config.reverse_class_map[class_num]] = class_num_to_count[class_num]

        cur_img_pred_class_counts = predictions["image_predictions"][img.img_name]["pred_class_counts"]

        for class_name in config.class_map.keys():
            annotated_img_counts[class_name].append(cur_img_class_counts[class_name])
            pred_img_counts[class_name].append(cur_img_pred_class_counts[class_name])

        #annotated_img_count = np.shape(img_abs_boxes)[0]

        #pred_img_count = predictions["image_predictions"][img.img_name]["pred_count"]
        pred_abs_boxes = np.array(predictions["image_predictions"][img.img_name]["nms_pred_img_abs_boxes"])
        pred_classes = np.array(predictions["image_predictions"][img.img_name]["nms_pred_classes"])
        pred_scores = np.array(predictions["image_predictions"][img.img_name]["nms_pred_scores"])

        pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                                                               img_abs_boxes, img_classes)
        img_metric_fn.add(pred_for_mAP, true_for_mAP)

        #annotated_img_counts.append(annotated_img_count)
        #pred_img_counts.append(pred_img_count)


    #annotated_img_counts = np.array(annotated_img_counts)
    #pred_img_counts = np.array(pred_img_counts)

    # annotated_patch_counts = []
    # pred_patch_counts = []

    annotated_patch_counts = {k: [] for k in config.class_map.keys()}
    pred_patch_counts = {k: [] for k in config.class_map.keys()}
    for patch_name, patch_pred in predictions["patch_predictions"].items():

        patch_classes = patch_pred["patch_classes"]
        unique, counts = np.unique(patch_classes, return_counts=True)
        class_num_to_count = dict(zip(unique, counts))
        cur_patch_class_counts = {k: 0 for k in config.class_map.keys()}
        for class_num in class_num_to_count.keys():
            cur_patch_class_counts[config.reverse_class_map[class_num]] = class_num_to_count[class_num]

        pred_patch_classes = patch_pred["pred_classes"]
        pred_unique, pred_counts = np.unique(pred_patch_classes, return_counts=True)
        class_num_to_pred_count = dict(zip(pred_unique, pred_counts))
        cur_patch_pred_class_counts = {k: 0 for k in config.class_map.keys()}
        for class_num in class_num_to_pred_count.keys():
            cur_patch_pred_class_counts[config.reverse_class_map[class_num]] = class_num_to_pred_count[class_num]

        for class_name in config.class_map.keys():
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


def gather_stats(pred_dir, img_set, img_nms_iou_thresh):

    logger = logging.getLogger(__name__)

    predictions = json_io.load_json(os.path.join(pred_dir, "predictions.json"))
    out_path = os.path.join(pred_dir, "prediction_stats.json")

    is_annotated = predictions["is_annotated"]

    eval_data = {}
    eval_data["is_annotated"] = is_annotated
    eval_data["patch_results"] = {}
    eval_data["image_results"] = {}
    eval_data["patch_summary_results"] = {}
    eval_data["img_summary_results"] = {}

    patch_pred_sum = 0
    img_pred_sum = 0

    if is_annotated:
        patch_actual_sum = 0
        img_actual_sum = 0


    patch_predictions = decode_predictions.decode_patch_predictions(predictions)
    img_predictions = decode_predictions.decode_img_predictions(predictions, img_set.class_map, img_nms_iou_thresh)

    patch_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=img_set.num_classes)
    img_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=img_set.num_classes)

    for patch_path, patch_pred in tqdm.tqdm(patch_predictions.items(), "Gathering prediction stats for patches"):
        

        num_pred = int((patch_pred["pred_patch_abs_boxes"]).size // 4)

        patch_pred_sum += num_pred
        eval_data["patch_results"][patch_path] = {
            "num_pred": num_pred,
        }
        
        if is_annotated:
            num_actual = int((patch_pred["patch_abs_boxes"]).size // 4)

            patch_actual_sum += num_actual
            pred_minus_actual = num_pred - num_actual
            abs_pred_minus_actual = abs(pred_minus_actual)


            if num_pred > 0 and num_actual > 0:
                pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
                                               patch_pred["pred_patch_abs_boxes"],
                                               patch_pred["pred_classes"],
                                               patch_pred["pred_scores"],
                                               patch_pred["patch_abs_boxes"], 
                                               patch_pred["patch_classes"])

                patch_metric_fn.add(pred_for_mAP, true_for_mAP)

            eval_data["patch_results"][patch_path].update({
                "num_actual": num_actual,
                "pred_minus_actual": pred_minus_actual,
                "abs_pred_minus_actual": abs_pred_minus_actual
            })



    for img_path, img_pred in tqdm.tqdm(img_predictions.items(), "Gathering prediction stats for images"):

        num_pred = int((img_pred["nms_pred_img_abs_boxes"]).size // 4)
        img_pred_sum += num_pred
        eval_data["image_results"][img_path] = {
            "num_pred": num_pred
        }

        if is_annotated:
            num_actual = int((img_pred["img_abs_boxes"]).size // 4)

            img_actual_sum += num_actual
            pred_minus_actual = num_pred - num_actual
            abs_pred_minus_actual = abs(pred_minus_actual)

            if num_pred > 0 and num_actual > 0:
                pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
                                               img_pred["nms_pred_img_abs_boxes"],
                                               img_pred["nms_pred_classes"],
                                               img_pred["nms_pred_scores"],
                                               img_pred["img_abs_boxes"], 
                                               img_pred["img_classes"])

                img_metric_fn.add(pred_for_mAP, true_for_mAP)

            eval_data["image_results"][img_path].update({
                "num_pred": num_pred,
                "num_actual": num_actual,
                "pred_minus_actual": pred_minus_actual,
                "abs_pred_minus_actual": abs_pred_minus_actual
            })


    logger.info("Calculating summary statistics ...")

    eval_data["patch_summary_results"]["pred_sum"] = patch_pred_sum
    eval_data["img_summary_results"]["pred_sum"] = img_pred_sum


    if is_annotated:
        eval_data["patch_summary_results"]["actual_sum"] = patch_actual_sum
        eval_data["img_summary_results"]["actual_sum"] = img_actual_sum

        eval_data["patch_summary_results"]["pred_minus_actual"] = \
                five_num_summary([eval_data["patch_results"][patch_path]["pred_minus_actual"] for patch_path in eval_data["patch_results"].keys()])

        patch_pascal_voc_score = patch_metric_fn.value(iou_thresholds=0.5)['mAP']
        patch_coco_score = patch_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        img_pascal_voc_score = img_metric_fn.value(iou_thresholds=0.5)['mAP']
        img_coco_score = img_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        eval_data["patch_summary_results"]["pascal_voc_score"] = float(patch_pascal_voc_score)
        eval_data["patch_summary_results"]["coco_score"] = float(patch_coco_score)

        eval_data["img_summary_results"]["pascal_voc_score"] = float(img_pascal_voc_score)
        eval_data["img_summary_results"]["coco_score"] = float(img_coco_score)


    logger.info("Finished gathering statistics.")

    json_io.save_json(out_path, eval_data)



