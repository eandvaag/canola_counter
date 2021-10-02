import numpy as np
import tqdm
import os
from mean_average_precision import MetricBuilder


import models.detectors.common.box_utils as box_utils
from io_utils import json_io



def get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                              true_abs_boxes, true_classes):

    pred_abs_boxes = box_utils.swap_xy_np(pred_abs_boxes)
    pred_classes = np.expand_dims(pred_classes, axis=-1)
    pred_scores = np.expand_dims(pred_scores, axis=-1)
    pred = np.hstack([pred_abs_boxes, pred_classes, pred_scores])

    true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    true_classes = np.expand_dims(true_classes, axis=-1)
    difficult = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    crowd = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    true = np.hstack([true_abs_boxes, true_classes, difficult, crowd])  

    return pred, true


def evaluate(model, patch_dir, settings):

    pred_dir = os.path.join(model.config.model_dir, os.path.basename(patch_dir))
    out_path = os.path.join(pred_dir, "prediction_stats.json")
    if os.path.exists(out_path):
        return

    eval_data = {}
    eval_data["patch_results"] = {}
    eval_data["image_results"] = {}

    model.generate_predictions(patch_dir, skip_if_found=True)
    pred_patch_data, is_annotated = json_io.read_patch_predictions(model, patch_dir, settings)
    pred_img_data, is_annotated = json_io.read_img_predictions(model, patch_dir, settings)

    if not is_annotated:
        raise RuntimeError("Cannot evaluate unannotated images.")

    patch_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=settings.num_classes)
    img_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=settings.num_classes)

    for pred in tqdm.tqdm(pred_patch_data, "Evaluating patch predictions"):
        num_pred = int((pred["pred_patch_abs_boxes"]).size // 4)
        num_actual = int((pred["patch_abs_boxes"]).size // 4)

        pred_minus_actual = num_pred - num_actual
        abs_pred_minus_actual = abs(pred_minus_actual)


        if num_pred > 0 and num_actual > 0:
            pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
                                           pred["pred_patch_abs_boxes"],
                                           pred["pred_classes"],
                                           pred["pred_scores"],
                                           pred["patch_abs_boxes"], 
                                           pred["patch_classes"])

            patch_metric_fn.add(pred_for_mAP, true_for_mAP)

        eval_data["patch_results"][pred["patch_path"]] = {

            "num_pred": num_pred,
            "num_actual": num_actual,
            "pred_minus_actual": pred_minus_actual,
            "abs_pred_minus_actual": abs_pred_minus_actual

        }

    for img_path in tqdm.tqdm(pred_img_data.keys(), "Evaluating image predictions"):

        num_pred = int((pred_img_data[img_path]["nms_pred_img_abs_boxes"]).size // 4)
        num_actual = int((pred_img_data[img_path]["img_abs_boxes"]).size // 4)

        pred_minus_actual = num_pred - num_actual
        abs_pred_minus_actual = abs(pred_minus_actual)

        if num_pred > 0 and num_actual > 0:
            pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
                                           pred_img_data[img_path]["nms_pred_img_abs_boxes"],
                                           pred_img_data[img_path]["nms_pred_classes"],
                                           pred_img_data[img_path]["nms_pred_scores"],
                                           pred_img_data[img_path]["img_abs_boxes"], 
                                           pred_img_data[img_path]["img_classes"])

            img_metric_fn.add(pred_for_mAP, true_for_mAP)

        eval_data["image_results"][img_path] = {

            "num_pred": num_pred,
            "num_actual": num_actual,
            "pred_minus_actual": pred_minus_actual,
            "abs_pred_minus_actual": abs_pred_minus_actual
        } 


    eval_data["patch_summary_results"] = {}
    eval_data["img_summary_results"] = {}

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


    json_io.save_json(out_path, eval_data)



def five_num_summary(data):
    return np.percentile(np.array(data), [0, 25, 50, 75, 100], interpolation='midpoint').tolist()